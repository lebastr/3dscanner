with import <nixpkgs> {};
let python_env = python3.withPackages (ps: with ps;
      let local_imgaug = callPackage ./imgaug.nix { };
          local_tensorboardx = callPackage ./tensorboardx.nix { };

      in [
      imageio
      jupyter
      local_imgaug
      local_tensorboardx
      matplotlib
      networkx
      numpy
      opencv3
      pillow
      pytorch
      scikitimage
      scipy
      tqdm
      ]);

    link = "python-env";
    shellHook = ''
         nix-store --add-root python-env --indirect -r ${python_env}
    '';

in python_env.env.overrideAttrs (x: { shellHook = shellHook; })
