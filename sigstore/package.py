# Copyright 2026 The Sigstore Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
API for signing and verifying Python packages.

This module supports the three standard Python distribution formats:

* Source distributions (sdists): `name-version.tar.gz` (PEP 625);
* Built distributions (wheels): `name-version-...-.whl` (PEP 427);
* Legacy zip-format source distributions: `name-version.zip`.

It is a thin convenience layer over the existing signing and verification
APIs: packages are signed with `sigstore.sign.Signer.sign_artifact` and
verified with `sigstore.verify.Verifier.verify_artifact`, producing and
consuming ordinary Sigstore `Bundle` objects.

Example:

```python
from pathlib import Path

from sigstore.models import Bundle, ClientTrustConfig
from sigstore.oidc import Issuer
from sigstore.package import Package, sign_package, verify_package
from sigstore.sign import SigningContext
from sigstore.verify import Verifier
from sigstore.verify.policy import Identity

package = Package.from_path(Path("dist/example-1.0.0-py3-none-any.whl"))

# Sign the package.
trust_config = ClientTrustConfig.production()
issuer = Issuer(trust_config.signing_config.get_oidc_url())
context = SigningContext.from_trust_config(trust_config)
token = issuer.identity_token()
with context.signer(token) as signer:
    bundle = sign_package(package, signer)

# Verify the package.
verifier = Verifier.production()
verify_package(
    package,
    bundle,
    Identity(identity="foo@bar.com", issuer="https://accounts.google.com"),
    verifier,
)
```
"""

from __future__ import annotations

import logging
import tarfile
import zipfile
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from sigstore._utils import sha256_digest
from sigstore.errors import PackageError
from sigstore.hashes import Hashed
from sigstore.models import Bundle
from sigstore.sign import Signer
from sigstore.verify.policy import VerificationPolicy
from sigstore.verify.verifier import Verifier

_logger = logging.getLogger(__name__)


class PackageType(StrEnum):
    """
    The kind of a Python package (distribution archive).
    """

    SDIST = "sdist"
    """
    A source distribution: a gzipped tarball named `name-version.tar.gz`.
    """

    WHEEL = "wheel"
    """
    A built distribution: a ZIP archive named
    `name-version-{tags}.whl`.
    """

    ZIP_SDIST = "zip-sdist"
    """
    A legacy zip-format source distribution: `name-version.zip`.
    """


def _parse_name_version(stem: str, path: Path) -> tuple[str, str]:
    """
    Split a `name-version` filename stem into its components.

    The version is the final `-`-separated component, per PEP 625.
    """
    name, sep, version = stem.rpartition("-")
    if not sep or not name or not version:
        raise PackageError(
            f"invalid Python package filename: {path.name} "
            "(expected 'name-version' format)"
        )
    return name, version


def _validate_wheel(path: Path) -> None:
    """
    Ensure that the wheel at `path` is a readable ZIP archive containing
    a `.dist-info/METADATA` entry.
    """
    try:
        with zipfile.ZipFile(path) as zf:
            has_metadata = any(
                member.count("/") == 1
                and member.endswith(".dist-info/METADATA")
                and not member.startswith("/")
                for member in zf.namelist()
            )
    except (OSError, zipfile.BadZipFile) as exc:
        raise PackageError(f"invalid wheel {path}: not a readable ZIP archive") from exc

    if not has_metadata:
        raise PackageError(
            f"invalid wheel {path}: no '.dist-info/METADATA' entry found"
        )


def _validate_sdist(path: Path, name: str, version: str) -> None:
    """
    Ensure that the sdist at `path` is a readable gzipped tarball containing
    a `PKG-INFO` entry in its top-level directory.
    """
    try:
        with tarfile.open(path, mode="r:gz") as tf:
            has_pkg_info = any(
                member.count("/") == 1 and member.endswith("/PKG-INFO")
                for member in tf.getnames()
            )
    except (OSError, tarfile.TarError) as exc:
        raise PackageError(
            f"invalid sdist {path}: not a readable gzipped tarball"
        ) from exc

    if not has_pkg_info:
        raise PackageError(
            f"invalid sdist {path}: no top-level 'PKG-INFO' entry found "
            f"(expected under '{name}-{version}/')"
        )


def _validate_zip_sdist(path: Path) -> None:
    """
    Ensure that the zip-format sdist at `path` is a readable ZIP archive
    containing a `PKG-INFO` entry in its top-level directory.
    """
    try:
        with zipfile.ZipFile(path) as zf:
            has_pkg_info = any(
                member.count("/") == 1 and member.endswith("/PKG-INFO")
                for member in zf.namelist()
            )
    except (OSError, zipfile.BadZipFile) as exc:
        raise PackageError(
            f"invalid zip sdist {path}: not a readable ZIP archive"
        ) from exc

    if not has_pkg_info:
        raise PackageError(
            f"invalid zip sdist {path}: no top-level 'PKG-INFO' entry found"
        )


@dataclass(frozen=True)
class Package:
    """
    A Python package (distribution archive) on disk, identified and
    validated for signing or verification.

    Use `Package.from_path` to construct an instance.
    """

    path: Path
    """
    The path to the package archive.
    """

    type: PackageType
    """
    The kind of distribution (sdist, wheel, or zip-format sdist).
    """

    name: str
    """
    The distribution's name, as parsed from its filename.
    """

    version: str
    """
    The distribution's version, as parsed from its filename.
    """

    @classmethod
    def from_path(cls, path: str | Path) -> Package:
        """
        Identify, parse, and validate the Python package at `path`.

        The package's type is detected from its filename extension:
        `.tar.gz` for sdists, `.whl` for wheels, and `.zip` for legacy
        zip-format sdists. The distribution's name and version are parsed
        from the filename, and the archive itself is checked for basic
        structural validity (i.e. that it is a readable archive containing
        the expected package metadata).

        Raises `PackageError` if the file is missing, has an unrecognized
        extension, has an unparsable filename, or is not a well-formed
        package archive.
        """
        path = Path(path)
        if not path.is_file():
            raise PackageError(f"package not found: {path}")

        filename = path.name
        if filename.endswith(".tar.gz"):
            package_type = PackageType.SDIST
            name, version = _parse_name_version(filename[: -len(".tar.gz")], path)
            _validate_sdist(path, name, version)
        elif filename.endswith(".whl"):
            package_type = PackageType.WHEEL
            # Wheel filenames are {name}-{version}(-{build})?-{python tag}
            # -{abi tag}-{platform tag}.whl; the name and version are the
            # first two components.
            components = filename[: -len(".whl")].split("-")
            if len(components) < 5 or not components[0] or not components[1]:
                raise PackageError(
                    f"invalid wheel filename: {filename} "
                    "(expected 'name-version-{tags}.whl' format)"
                )
            name, version = components[0], components[1]
            _validate_wheel(path)
        elif filename.endswith(".zip"):
            package_type = PackageType.ZIP_SDIST
            name, version = _parse_name_version(filename[: -len(".zip")], path)
            _validate_zip_sdist(path)
        else:
            raise PackageError(
                f"unrecognized Python package extension: {filename} "
                "(expected one of: .tar.gz, .whl, .zip)"
            )

        return cls(path=path, type=package_type, name=name, version=version)

    def hashed(self) -> Hashed:
        """
        Compute the SHA256 digest of the package's contents, streaming the
        archive from disk rather than buffering it in memory.
        """
        with self.path.open("rb") as io:
            return sha256_digest(io)


def _as_package(package: Package | str | Path) -> Package:
    """
    Coerce the given value into a validated `Package`.
    """
    if isinstance(package, Package):
        return package
    return Package.from_path(package)


def sign_package(package: Package | str | Path, signer: Signer) -> Bundle:
    """
    Sign the given Python package with `signer`, returning a Sigstore
    `Bundle` containing the signing materials.

    `package` is either a `Package` (as produced by `Package.from_path`)
    or a path to a package archive, in which case it is identified and
    validated first.

    `signer` is a `sigstore.sign.Signer`, as produced by
    `SigningContext.signer`. The package is signed exactly like any other
    artifact: the resulting bundle contains a `hashedrekord` entry over
    the package's SHA256 digest.
    """
    package = _as_package(package)
    _logger.debug(f"signing {package.type.value} package: {package.path}")

    return signer.sign_artifact(package.hashed())


def verify_package(
    package: Package | str | Path,
    bundle: Bundle,
    policy: VerificationPolicy,
    verifier: Verifier,
) -> None:
    """
    Verify the given Python package against a Sigstore `Bundle`.

    `package` is either a `Package` (as produced by `Package.from_path`)
    or a path to a package archive, in which case it is identified and
    validated first.

    `bundle` is the Sigstore `Bundle` to verify against, as produced by
    `sign_package` (or `Signer.sign_artifact`).

    `policy` is the `VerificationPolicy` to verify the signing certificate
    against, and `verifier` is a `sigstore.verify.Verifier`.

    On failure, this function raises `VerificationError`.
    """
    package = _as_package(package)
    _logger.debug(f"verifying {package.type.value} package: {package.path}")

    verifier.verify_artifact(package.hashed(), bundle, policy)
