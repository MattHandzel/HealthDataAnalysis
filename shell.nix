{
  pkgs ? import <nixpkgs> {} }:
  pkgs.mkShell {
    buildInputs = with pkgs; [
      # Add your dependencies here

python3Packages.pandas
python3Packages.numpy
python3Packages.ics
python3Packages.matplotlib
python3Packages.seaborn
python3Packages.pyperclip
python3Packages.loguru
    ];


  }
