import os
import subprocess
import sys

def install_package(package_path, package_dir):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_path, "--no-index", "--find-links", package_dir])
    except subprocess.CalledProcessError as e:
        print(f"Failed to install {package_path}: {e}")
        return False
    return True

def install_all_packages(package_dir):
    # Get a list of all package files in the directory
    packages = [os.path.join(package_dir, f) for f in os.listdir(package_dir) if f.endswith(('.whl', '.tar.gz', '.zip'))]

    installed = set()
    while packages:
        remaining = []
        for package in packages:
            if install_package(package, package_dir):
                installed.add(package)
            else:
                remaining.append(package)
        if len(remaining) == len(packages):
            print("Could not resolve some dependencies. Remaining packages:")
            for pkg in remaining:
                print(pkg)
            break
        packages = remaining

if __name__ == "__main__":
    package_dir = "packages"  # Directory containing all your package files
    install_all_packages(package_dir)