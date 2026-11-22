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

import hashlib
import io
import tarfile
import zipfile
from pathlib import Path

import pretend
import pytest

from sigstore.errors import PackageError, VerificationError
from sigstore.hashes import Hashed
from sigstore.package import Package, PackageType, sign_package, verify_package

NAME = "example"
VERSION = "1.0.0"


def _make_sdist(path: Path, *, with_pkg_info: bool = True, extra: bool = False) -> Path:
    """Build a minimal `.tar.gz` sdist."""
    pkg_info = b"Metadata-Version: 2.1\nName: example\nVersion: 1.0.0\n"
    with tarfile.open(path, mode="w:gz") as tf:
        if with_pkg_info:
            info = tarfile.TarInfo(f"{NAME}-{VERSION}/PKG-INFO")
            info.size = len(pkg_info)
            tf.addfile(info, io.BytesIO(pkg_info))
        if extra:
            payload = b"tampered"
            info = tarfile.TarInfo(f"{NAME}-{VERSION}/extra.txt")
            info.size = len(payload)
            tf.addfile(info, io.BytesIO(payload))
    return path


def _make_wheel(path: Path, *, with_metadata: bool = True, extra: bool = False) -> Path:
    """Build a minimal `.whl` archive."""
    with zipfile.ZipFile(path, mode="w") as zf:
        if with_metadata:
            zf.writestr(
                f"{NAME}-{VERSION}.dist-info/METADATA",
                "Metadata-Version: 2.1\nName: example\nVersion: 1.0.0\n",
            )
        zf.writestr(f"{NAME}/__init__.py", "")
        if extra:
            zf.writestr(f"{NAME}/extra.txt", "tampered")
    return path


def _make_zip_sdist(
    path: Path, *, with_pkg_info: bool = True, extra: bool = False
) -> Path:
    """Build a minimal `.zip` sdist."""
    with zipfile.ZipFile(path, mode="w") as zf:
        if with_pkg_info:
            zf.writestr(
                f"{NAME}-{VERSION}/PKG-INFO",
                "Metadata-Version: 2.1\nName: example\nVersion: 1.0.0\n",
            )
        if extra:
            zf.writestr(f"{NAME}-{VERSION}/extra.txt", "tampered")
    return path


@pytest.fixture
def sdist(tmp_path: Path) -> Path:
    return _make_sdist(tmp_path / f"{NAME}-{VERSION}.tar.gz")


@pytest.fixture
def wheel(tmp_path: Path) -> Path:
    return _make_wheel(tmp_path / f"{NAME}-{VERSION}-py3-none-any.whl")


@pytest.fixture
def zip_sdist(tmp_path: Path) -> Path:
    return _make_zip_sdist(tmp_path / f"{NAME}-{VERSION}.zip")


class TestPackageFromPath:
    @pytest.mark.parametrize(
        ("fixture", "expected_type"),
        [
            ("sdist", PackageType.SDIST),
            ("wheel", PackageType.WHEEL),
            ("zip_sdist", PackageType.ZIP_SDIST),
        ],
    )
    def test_detects_package_types(self, request, fixture, expected_type):
        package = Package.from_path(request.getfixturevalue(fixture))

        assert package.type == expected_type
        assert package.name == NAME
        assert package.version == VERSION

    @pytest.mark.parametrize("fixture", ["sdist", "wheel", "zip_sdist"])
    def test_accepts_str_paths(self, request, fixture):
        path = request.getfixturevalue(fixture)
        package = Package.from_path(str(path))

        assert package.path == path

    @pytest.mark.parametrize("fixture", ["sdist", "wheel", "zip_sdist"])
    def test_hashed_matches_file_digest(self, request, fixture):
        path = request.getfixturevalue(fixture)
        package = Package.from_path(path)

        assert package.hashed().digest == hashlib.sha256(path.read_bytes()).digest()

    def test_wheel_with_build_tag(self, tmp_path):
        path = _make_wheel(tmp_path / f"{NAME}-{VERSION}-1-py3-none-any.whl")
        package = Package.from_path(path)

        assert package.type == PackageType.WHEEL
        assert package.name == NAME
        assert package.version == VERSION

    def test_missing_file(self, tmp_path):
        with pytest.raises(PackageError, match="package not found"):
            Package.from_path(tmp_path / "example-1.0.0.tar.gz")

    def test_unrecognized_extension(self, tmp_path):
        path = tmp_path / "example-1.0.0.tar.bz2"
        path.write_bytes(b"not a package")

        with pytest.raises(PackageError, match="unrecognized Python package"):
            Package.from_path(path)

    @pytest.mark.parametrize("suffix", [".tar.gz", ".zip"])
    def test_unparsable_filename(self, tmp_path, suffix):
        path = tmp_path / f"noversion{suffix}"
        path.write_bytes(b"not a package")

        with pytest.raises(PackageError, match="invalid Python package filename"):
            Package.from_path(path)

    def test_invalid_wheel_filename(self, tmp_path):
        path = tmp_path / "example.whl"
        path.write_bytes(b"not a package")

        with pytest.raises(PackageError, match="invalid wheel filename"):
            Package.from_path(path)

    @pytest.mark.parametrize(
        "filename",
        [
            "example-1.0.0.tar.gz",
            "example-1.0.0-py3-none-any.whl",
            "example-1.0.0.zip",
        ],
    )
    def test_corrupt_archives(self, tmp_path, filename):
        path = tmp_path / filename
        path.write_bytes(b"not an archive")

        with pytest.raises(PackageError, match="not a readable"):
            Package.from_path(path)

    def test_sdist_without_pkg_info(self, tmp_path):
        path = _make_sdist(tmp_path / f"{NAME}-{VERSION}.tar.gz", with_pkg_info=False)

        with pytest.raises(PackageError, match="PKG-INFO"):
            Package.from_path(path)

    def test_wheel_without_metadata(self, tmp_path):
        path = _make_wheel(
            tmp_path / f"{NAME}-{VERSION}-py3-none-any.whl", with_metadata=False
        )

        with pytest.raises(PackageError, match="METADATA"):
            Package.from_path(path)

    def test_zip_sdist_without_pkg_info(self, tmp_path):
        path = _make_zip_sdist(tmp_path / f"{NAME}-{VERSION}.zip", with_pkg_info=False)

        with pytest.raises(PackageError, match="PKG-INFO"):
            Package.from_path(path)


class TestSignPackage:
    @pytest.mark.parametrize("fixture", ["sdist", "wheel", "zip_sdist"])
    def test_signs_package_digest(self, request, fixture):
        path = request.getfixturevalue(fixture)
        package = Package.from_path(path)
        expected = hashlib.sha256(path.read_bytes()).digest()

        bundle = pretend.stub()
        sign_artifact = pretend.call_recorder(lambda input_: bundle)
        signer = pretend.stub(sign_artifact=sign_artifact)

        result = sign_package(package, signer)

        assert result is bundle
        assert len(sign_artifact.calls) == 1
        hashed = sign_artifact.calls[0].args[0]
        assert isinstance(hashed, Hashed)
        assert hashed.digest == expected

    @pytest.mark.parametrize("fixture", ["sdist", "wheel", "zip_sdist"])
    def test_accepts_paths(self, request, fixture):
        path = request.getfixturevalue(fixture)
        bundle = pretend.stub()
        signer = pretend.stub(sign_artifact=pretend.call_recorder(lambda _: bundle))

        assert sign_package(path, signer) is bundle

    def test_invalid_package_raises_before_signing(self, tmp_path):
        signer = pretend.stub(
            sign_artifact=pretend.raiser(AssertionError("must not be called"))
        )

        with pytest.raises(PackageError):
            sign_package(tmp_path / "missing-1.0.0.tar.gz", signer)


class TestVerifyPackage:
    @pytest.mark.parametrize("fixture", ["sdist", "wheel", "zip_sdist"])
    def test_verifies_package_digest(self, request, fixture, null_policy):
        path = request.getfixturevalue(fixture)
        package = Package.from_path(path)
        expected = hashlib.sha256(path.read_bytes()).digest()

        bundle = pretend.stub()
        verify_artifact = pretend.call_recorder(lambda *args: None)
        verifier = pretend.stub(verify_artifact=verify_artifact)

        verify_package(package, bundle, null_policy, verifier)

        assert len(verify_artifact.calls) == 1
        call = verify_artifact.calls[0]
        hashed, called_bundle, called_policy = call.args
        assert isinstance(hashed, Hashed)
        assert hashed.digest == expected
        assert called_bundle is bundle
        assert called_policy is null_policy

    @pytest.mark.parametrize("fixture", ["sdist", "wheel", "zip_sdist"])
    def test_accepts_paths(self, request, fixture, null_policy):
        path = request.getfixturevalue(fixture)
        verifier = pretend.stub(
            verify_artifact=pretend.call_recorder(lambda *args: None)
        )

        verify_package(path, pretend.stub(), null_policy, verifier)

        assert len(verifier.verify_artifact.calls) == 1

    @pytest.mark.parametrize("fixture", ["sdist", "wheel", "zip_sdist"])
    def test_propagates_verification_failure(self, request, fixture, null_policy):
        path = request.getfixturevalue(fixture)
        verifier = pretend.stub(
            verify_artifact=pretend.raiser(VerificationError("invalid signature"))
        )

        with pytest.raises(VerificationError, match="invalid signature"):
            verify_package(path, pretend.stub(), null_policy, verifier)

    def test_invalid_package_raises_before_verifying(self, tmp_path, null_policy):
        verifier = pretend.stub(
            verify_artifact=pretend.raiser(AssertionError("must not be called"))
        )

        with pytest.raises(PackageError):
            verify_package(
                tmp_path / "missing-1.0.0.tar.gz",
                pretend.stub(),
                null_policy,
                verifier,
            )


@pytest.mark.parametrize("fixture", ["sdist", "wheel", "zip_sdist"])
def test_sign_verify_round_trip(request, fixture, null_policy, tmp_path):
    """
    A bundle produced by `sign_package` for a package is verified by
    `verify_package` against the same package contents, and rejected
    for tampered contents.
    """
    path = request.getfixturevalue(fixture)
    package = Package.from_path(path)

    # A fake signer that behaves like `Signer.sign_artifact`: it records
    # the digest it signed and returns it as the "bundle".
    signer = pretend.stub(
        sign_artifact=pretend.call_recorder(lambda hashed: ("bundle", hashed.digest))
    )
    bundle = sign_package(package, signer)

    # A fake verifier that behaves like `Verifier.verify_artifact`: it
    # checks the verified digest against the one recorded in the bundle.
    def _verify_artifact(hashed, bundle, policy):
        _, expected_digest = bundle
        if hashed.digest != expected_digest:
            raise VerificationError("digest mismatch")

    verifier = pretend.stub(verify_artifact=_verify_artifact)

    # Same package contents: verification succeeds.
    verify_package(package, bundle, null_policy, verifier)

    # Tampered package contents: verification fails.
    tampered = tmp_path / f"tampered-{path.name}"
    if package.type == PackageType.SDIST:
        _make_sdist(tampered, extra=True)
    elif package.type == PackageType.WHEEL:
        _make_wheel(tampered, extra=True)
    else:
        _make_zip_sdist(tampered, extra=True)

    with pytest.raises(VerificationError, match="digest mismatch"):
        verify_package(Package.from_path(tampered), bundle, null_policy, verifier)
