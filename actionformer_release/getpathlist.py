import os



output_file = 'feat_file_list.txt'

with open(json_file, 'r') as fid:
    json_data = json.load(fid)

json_db = json_data

with open(output_file, 'w') as f:
    for key, value in json_db.items():
        # ファイル名を生成
        feat_file = os.path.join(feat_folder, file_prefix + key + file_ext)
        # テキストファイルに書き込む
        f.write(feat_file + '\n')
