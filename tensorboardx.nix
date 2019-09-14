{lib, fetchFromGitHub, buildPythonPackage, matplotlib, numpy, pillow, pytorch, protobuf, six, pytest, torchvision }:

buildPythonPackage rec {
  pname = "tensorboardx";
  version = "1.6";

  src = fetchFromGitHub {
    owner = "lanpa";
    repo = "tensorboardX";
    rev = "v${version}";
    sha256 = "0bxs9s686nram0dqhrkh21b8kq8lg7q3hf6v32pl51cvz2jvps8k";
  };

  checkInputs = [ matplotlib pillow pytorch pytest torchvision ];

  propagatedBuildInputs = [ numpy protobuf six ];

  preInstall=''
    rm tests/test_caffe2.py      # depends on unpackaged past library
  '';

  meta = {
    description = "Library for writing tensorboard-compatible logs";
    homepage = https://github.com/lanpa/tensorboard-pytorch;
    license = lib.licenses.mit;
    maintainers = with lib.maintainers; [ lebastr akamaus ];
  };
}
