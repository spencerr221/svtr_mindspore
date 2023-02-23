import os, zipfile, io, string


# def download_zip_file(zip_file_url, dst_dir):
#     os.makedirs(dst_dir, exist_ok=True)
#     r = requests.get(zip_file_url)
#     z = zipfile.ZipFile(io.BytesIO(r.content))
#     z.extractall(dst_dir)
#
#
# def unzip_file(zip_file_path, dst_dir=None):
#     with zipfile.ZipFile(zip_file_path, 'r') as zip:
#         # printing all the contents of the zip file
#         zip.printdir()
#
#         # extracting all the files
#         print('Extracting all the files now...')
#         if dst_dir:
#             zip.extractall(dst_dir)
#         else:
#             zip.extractall()
#         print('Done!')


def normalize_text(text):
    text = ''.join(
        filter(lambda x: x in (string.digits + string.ascii_letters), text))
    return text.lower()