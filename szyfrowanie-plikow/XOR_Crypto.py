import os
from itertools import cycle
import argparse


def encrypt(filepath: str, password: str, offset_range: tuple[int, int], inplace=False) -> None:
    try:
        if "\\" in filepath:
            savepath = "\\".join(filepath.split("\\")[:-1])
            with open(rf"{filepath}", "rb") as f:
                ext = filepath.split("\\")[-1].split(".")[-1]
                filename = filepath.split("\\")[-1].split(".")[0]
                data = f.read()
        elif "/" in filepath:
            savepath = "/".join(filepath.split("/")[:-1])
            with open(filepath, "rb") as f:
                ext = filepath.split("/")[-1].split(".")[-1]
                filename = filepath.split("/")[-1].split(".")[0]
                data = f.read()
        elif ("/" or "\\") not in filepath:
            savepath = f"./"
            with open(f"./{filepath}", "rb") as f:
                ext = filepath.split("/")[-1].split(".")[-1]
                filename = filepath.split("/")[-1].split(".")[0]
                data = f.read()            
    except FileNotFoundError:
        print(f"Provided file at {filepath} does not exist. Check and re-run with proper file location.")
        exit(-1)
    if inplace:
        os.remove(filepath)
    encrypted = bytes(plain ^ (pswrd + offset) for plain, pswrd, offset in
                      zip(data, cycle(bytes(password, "utf-8")), cycle(range(*offset_range))))
    with open(f"{savepath}/{filename}.{ext}.encr", "wb") as e:
        e.write(encrypted)


def decrypt(filepath: str, password: str, offset_range: tuple[int, int]) -> None:
    try:
        if "\\" in filepath:
            savepath = "\\".join(filepath.split("\\")[:-1])
            with open(rf"{filepath}", "rb") as f:
                filename = '.'.join(filepath.split("\\")[-1].split(".")[:-1])
                data = f.read()
        elif "/" in filepath:
            savepath = "/".join(filepath.split("/")[:-1])
            with open(filepath, "rb") as f:
                filename = '.'.join(filepath.split("/")[-1].split(".")[:-1])
                data = f.read()
        elif ("/" or "\\") not in filepath:
            savepath = f"./"
            with open(f"./{filepath}", "rb") as f:
                filename = '.'.join(filepath.split("/")[-1].split(".")[:-1])
                data = f.read()
    except FileNotFoundError:
        print(f"Provided file at {filepath} does not exist. Check and re-run with proper file location.")
        exit(-1)
    decrypted = bytes(plain ^ (pswrd + offset) for plain, pswrd, offset in
                      zip(data, cycle(bytes(password, "utf-8")), cycle(range(*offset_range))))
    with open(f"{savepath}/{filename}", "wb") as d:
        d.write(decrypted)
    os.remove(filepath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--encrypt", action="store_true", help="Encrypt mode toggle")
    parser.add_argument("-d", "--decrypt", action="store_true", help="Decrypt mode toggle")
    parser.add_argument("-p", "--password", required=True, help="Encryption/Decryption password")
    parser.add_argument("-fp", "--file_path", required=True, help="Path to file that needs to be encrypted/decrypted")
    parser.add_argument("-os", "--offset_start", required=True, help="Start of the offset, needs to be an Integer")
    parser.add_argument("-oe", "--offset_end", required=True, help="End of the offset, needs to be an Integer")
    parser.add_argument("-i", "--in_place", action="store_true", help="In-place encryption toggle. USE AT YOUR OWN RISK, DESTRUCTIVE")
    parser.add_argument("--in_place_no_prompt", action="store_true", help="OMITS IN PLACE PROMPT, VERY DANGEROUS AND DESTRUCTIVE")
    args = parser.parse_args()
    if args.encrypt:
        print("Encrypt mode engaged")
        if args.in_place:
            print(
                "In-place mode active. This is destructive and requires the parameters for offset and password to be remembered, else data will be lost.")
            print(f"Password: {args.password} Offset Start: {args.offset_start} Offset End:{args.offset_end}")
            while True:
                try:
                    cont = input("Continue? [Y/n] >> ")
                    if (not cont) or (cont in ["y", "Y"]):
                        print(f"Encrypting {args.file_path} in in-place mode.")
                        encrypt(args.file_path, args.password, (int(args.offset_start), int(args.offset_end)),
                                inplace=True)
                        break
                    elif cont in ["n", "N"]:
                        print(
                            f"Encrypting {args.file_path} non-destructively. This can be done without the -i or --in_place flag to omit this message")
                        encrypt(args.file_path, args.password, (int(args.offset_start), int(args.offset_end)))
                        break
                    else:
                        raise ValueError
                except ValueError:
                    print("Y/N answer expected with Y being default. Input proper answer.")
        else:
            if args.in_place_no_prompt:
                print(f"Encrypting {args.file_path} in-place.")
                encrypt(args.file_path, args.password, (int(args.offset_start), int(args.offset_end)), inplace=True)
            if not args.in_place_no_prompt:
                print(f"Encrypting {args.file_path}")
                encrypt(args.file_path, args.password, (int(args.offset_start), int(args.offset_end)))
    if args.decrypt:
        print(f"Decrypting {args.file_path}")
        decrypt(args.file_path, args.password, (int(args.offset_start), int(args.offset_end)))
