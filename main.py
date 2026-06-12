import os

from src.processing.rinex_sp3_merger import merge_rinex_sp3


def main() -> None:
    data_folder = "data"
    sp3_file = ""
    rinex_file = ""

    for f in os.listdir(data_folder):
        if f.lower().endswith(".sp3"):
            sp3_file = os.path.join(data_folder, f)
        if f.lower().endswith(".rnx"):
            rinex_file = os.path.join(data_folder, f)

    if not sp3_file or not rinex_file:
        print("Hata: Data klasorunde hem .sp3 hem .rnx dosyasi olmali!")
        return

    merge_rinex_sp3(sp3_file, rinex_file)


if __name__ == "__main__":
    main()
