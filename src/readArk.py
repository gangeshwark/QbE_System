import sys
import ark


def read_scp(scp_file):
    ark_reader = ark.ArkReader(scp_file)
    for key in ark_reader.scp_data:
        in_matr = ark_reader.read_utt_data(key)
        return in_matr


if __name__ == '__main__':
    if (len(sys.argv) < 2):
        print("USAGE: python " + sys.argv[0] + " scpFile")
        exit(1)
    scp = sys.argv[1]
    read_scp(scp)
