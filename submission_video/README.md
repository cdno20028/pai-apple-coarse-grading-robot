# 提出用完成動画の再現

このフォルダは、**P104/P105の実験そのものではなく、提出用に見せる完成動画の再現**のための追加資料です。

## 何が入っているか

- `p105_compose.py`  
  P105で得られた3本の実録動画と判定JSONを使い、提出用の完成動画を生成するスクリプトです。
- `p98r2_background.png`  
  工程98R2で承認済みの固定背景画像です。提出用画面の背景として使います。
- `p105_layout_bgfix.png`  
  背景修正版のレイアウト確認画像です。

## これは何のためか

第三者が **物理実験の再現** だけでなく、**提出時に見せた完成動画の見た目まで再現**できるようにするためです。

## 実行前提

先に `reproduction/p105_camera.py` を実行し、少なくとも次の4ファイルを得てください。

- `stage69_one_world_continuous_1x.mp4`
- `p105_grip_live.mp4`
- `p105_slider_follow_live.mp4`
- `P105_CAPTURE_VERDICT.json`

## 生成コマンド

```bash
python submission_video/p105_compose.py   --physics /path/to/stage69_one_world_continuous_1x.mp4   --grip /path/to/p105_grip_live.mp4   --follow /path/to/p105_slider_follow_live.mp4   --verdict /path/to/P105_CAPTURE_VERDICT.json   --photo submission_video/p98r2_background.png   --out PAI_final_p105.mp4
```

## 重要

- 合成は **P105の新規実行で得た実録動画だけ** を使います。
- 旧 `PAI_final_video.mp4` や旧試行の映像をつないで代用しません。
- 背景画像は提出用の見せ方のためのもので、物理実験そのものの入力ではありません。
- したがって、**物理実験の再現だけならこの背景画像は不要**ですが、**提出動画の完全再現には必要**です。
