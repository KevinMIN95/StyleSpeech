import argparse
import preprocessors.libritts as libritts

def main(data_path, sr):
    libritts.prepare_align_and_resample(data_path, sr)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='dataset/')
    parser.add_argument('--resample_rate', '-sr', type=int, default=16000)

    args = parser.parse_args()

    main(args.data_path, args.resample_rate)
