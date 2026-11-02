# Physical AI 最終課題｜第三者向けColab一括実行

提出用更新：2026/11/01 04:54:20 JST

1. GitHub上の `RUN_ALL_COLAB.ipynb` をColabで開く（またはZIP展開後にColabへアップロード）。
2. 最初のセルの `REPO_URL` に自分のPublic GitHubリポジトリURLを設定する。設定しない場合は `g108_public.zip` を一度アップロードする。
3. ［ランタイム］→［すべてのセルを実行］を押す。
4. `P107_NEW_RUN_RESULT PASS` と新しく生成された `P105_CAPTURE_VERDICT.json` を確認し、`P105_CAMERA_VIEWS.zip` を取得する。

ノートブックは既存 `reproduction/p105_camera.py` を変更せずに呼び出すだけである。P104とP105を二重実行しない。

### 合格条件

- 新規PyBullet実行で10/10、`reproduction_claim=true`、実行RC=0
- P105追加2カメラ各656フレーム、実際の`getCameraImage`、追従位置移動、撮影エラーなし
- 出力ZIPのCRC正常

### 禁止事項・限界

- 既存のP104/P105証拠ファイルを「今回の新規実行結果」として扱わない。
- 同じ作業フォルダに再実行しない。途中停止した場合はログを保全する。
- 写真17枚とUV等の実験素材は同梱しているが、未知果実の精度や実機安全を保証しない。
- 成功済みコードと写真・テクスチャのSHAは変更していない。
- Google Colabのパッケージ提供状況によってPython3.13用PyBulletのソースビルドが必要になることがある。これを含む新規Colabのエンドツーエンド完走試験はまだ行っていない。

### GitHubへ登録するもの

`g108_public.zip`は配布用アーカイブ。GitHub PublicリポジトリにはZIPを展開した**フォルダ内のすべてのファイル**を登録する。画像原本は `assets/*.part01` 等の分割ファイルから復元される。


### 提出用完成動画まで再現する場合

P107の物理実行とP105カメラ撮影が成功したら、得られた3本のMP4と `P105_CAPTURE_VERDICT.json` を使って、次を実行する。

```bash
python submission_video/p105_compose.py \
  --physics /path/to/stage69_one_world_continuous_1x.mp4 \
  --grip /path/to/p105_grip_live.mp4 \
  --follow /path/to/p105_slider_follow_live.mp4 \
  --verdict /path/to/P105_CAPTURE_VERDICT.json \
  --photo submission_video/p98r2_background.png \
  --out PAI_final_p105.mp4
```

これで、提出用に使用した完成動画の見た目まで第三者が再現できる。


### 工程108版のアップロード

`RUN_ALL_COLAB.ipynb` は `g108_ready.zip` と `g108.zip` を受け付ける。すでにP107で合格した場合、撮影の再実行は不要。動画の合成は `submission_video/README.md` に従い別途行う。
