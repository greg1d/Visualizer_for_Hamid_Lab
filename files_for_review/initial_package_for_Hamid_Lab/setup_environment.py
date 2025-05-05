import subprocess
import sys


def install_with_pip(package):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        return False


def is_installed_with_pip(package):
    try:
        __import__(package)
        return True
    except ImportError:
        return False


def install(package):
    if not install_with_pip(package):
        return False
    return True


def check_and_install_packages(requirements_file):
    with open(requirements_file, "r") as file:
        required_packages = file.read().splitlines()

    installed_any = False
    installed_packages = []
    for package in required_packages:
        if is_installed_with_pip(package):
            print(f"{package} is already installed.")
        else:
            if install(package):
                installed_any = True
                installed_packages.append(package)

    if installed_any:
        print("#" * 100)
        print("Installed packages:")
        for package in installed_packages:
            print(package)
        print("#" * 100)
    else:
        print("#" * 100)
        print("Nothing to add - you're all good")
        print("#" * 100)


if __name__ == "__main__":
    requirements_file = "requirements.txt"
    check_and_install_packages(requirements_file)
