import os
import sys
import shutil
import argparse


def download_yt(video_fp, out_root, overwrite):
    """
    url_file: file path to simple text file containing list of youtube video urls. One video per line
    out_root: root directory of all output, such as video files and frames
    overwrite: boolean whether to overwrite the current video directory if it exists

    saves the youtube video as a video file and
    returns the youtube video file path
    """
    
    video_dir = os.path.join(out_root, 'test_videos')
    make_dir(video_dir, overwrite)
    
    os.system('yt-dlp -o '+ os.path.join(video_dir, '%\(id\)s.%\(ext\)s') + ' -f bv*+ba[height=480]/bv*+ba' + ' --batch-file ' + video_fp)
    
    return [os.path.join(video_dir, f) for f in os.listdir(video_dir) if not os.path.isdir(f)]



def make_dir(path, overwrite):
    if overwrite:
        try:
            shutil.rmtree(path)
        except:
            pass
    if not os.path.isdir(path):
        os.mkdir(path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('vid_location_file', type=str)
    parser.add_argument('out_root', type=str)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args(sys.argv[1:])
    
    print(args)

    make_dir(args.out_root, args.overwrite)

    vid_files = download_yt(
        args.vid_location_file,
        args.out_root,
        args.overwrite
    )
    

if __name__ == '__main__':
    main()
