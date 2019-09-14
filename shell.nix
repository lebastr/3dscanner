with import <nixpkgs> {};
let python_env = python3.withPackages (ps: with ps;
      let local_imgaug = callPackage ./imgaug.nix { };
      in [ numpy matplotlib jupyter pillow opencv3
           scipy imageio local_imgaug
           scikitimage pytorch tqdm networkx ]);

    link = "python-env";
    shellHook = ''
         nix-store --add-root python-env --indirect -r ${python_env}
    '';

in python_env.env.overrideAttrs (x: { shellHook = shellHook; })
