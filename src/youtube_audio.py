from yt_dlp import YoutubeDL
import os
import re

class YouTubeAudioDownloader:
    
    def __init__(self, output_folder="./data/audios"):
        self.output_folder = os.path.abspath(output_folder)
        self.audio_files_dict = {}

    def get_safe_filename(self, filename):
        safe_filename = re.sub(r'[^\w\-.]', '_', filename)
        safe_filename = re.sub(r'_+', '_', safe_filename)
        safe_filename = safe_filename[:50].strip('_')
        return safe_filename

    def download_audio(self, video_url):
        """

        preferredquality - set to values 128kbps, 192kbps, 256kbps, 320kbps

        """
        try:
            ydl_opts = {
                'format': 'bestaudio/best',
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '192',
                }],
                'outtmpl': os.path.join(self.output_folder, '%(title)s.%(ext)s'),
            }

            with YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(video_url, download=True)
                filename = ydl.prepare_filename(info)
                base, ext = os.path.splitext(filename)
                new_file = base + '.mp3'

            print(f"Audio file downloaded: {new_file}")
            self.audio_files_dict[video_url] = new_file
            return new_file
        except Exception as e:
            print(f"Error downloading audio from {video_url}: {str(e)}")
            return None

    def download_multiple_audios(self, video_urls):
        for url in video_urls:
            print(f"Processing video: {url}")
            audio_file = self.download_audio(url)
            if audio_file is None:
                print(f"Failed to download audio from video: {url}")
        return self.audio_files_dict

if __name__ == "__main__":
    # Example usage:
    video_urls = [
        "https://www.youtube.com/watch?v=sNa_uiqSlJo",
        "https://www.youtube.com/watch?v=OnIQrDiTtRM",
        "https://www.youtube.com/watch?v=6qCrvlHRhcM",
    ]
    output_folder = "./data/audios"
    downloader = YouTubeAudioDownloader(output_folder)
    audio_files_dict = downloader.download_multiple_audios(video_urls)

    for video_url, audio_file in audio_files_dict.items():
        print(f"Audio file for {video_url}: {audio_file}")
