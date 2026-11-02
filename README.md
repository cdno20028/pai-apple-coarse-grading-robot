# Apple Physical AI — Physical AI 応用編1 最終課題 / 公開・再現資料

提出用更新：2026/11/01 04:54:20 JST

## 最短の第三者再現方法

**Colabで `RUN_ALL_COLAB.ipynb` を開き、［ランタイム］→［すべてのセルを実行］。** 最初に `g108_public.zip` をアップロードするか、冒頭セルの `REPO_URL` にGitHub PublicリポジトリURLを設定すればGitHubから直接取得できる。あとは、原本ZIPの復元・静的検証・依存導入・**P104の物理10件とP105の実録カメラを含む一度だけの実行**・新規結果判定まで一続きである。

- 旧P106の**実行コード・画像原本・監査データ223ファイルはそのまま**。変更はこのREADMEと、新しいノートブック・手順書・静的確認スクリプトのみ。
- 物理制御やAIを修正していない。既知素材での10/10成功、2カメラ各656フレームは**保存済みの成功記録**である。さらに2026年11月1日、独立Colab実行者の新規結果でP107一括実行の10/10・2視点撮影の完走も確認した。
- 失敗したときは途中成果物を確認し、再実行を自動的に行わない。
- ノートブックを使わず個別に進める手順は下に残した。

---

# Apple Physical AI — 公開用コードと再現データ（P106）

資料更新：2026/10/31 18:22:41 JST

松尾研 Physical AI 応用編1 最終課題：畑でのりんご粗選果を想定したPyBullet 3本指ロボット。外観画像からAIがA/B/C/Dを判定し、満杯時は停止して作業者回収の模擬イベントを待ち、異物は把持しない。

## 実際に検証済みの結果

- **P104：単一の連続PyBullet世界で10ケース中10ケース合格。** りんご9件の等級別搬送とテニスボール1件の拒否。A枠満杯停止、模擬回収後の再開を含む。
- **P105：P104のAI・物理制御を変えずに、把持アップと3Dスライダー追従カメラをそれぞれ656フレーム、20fps、32.8秒撮影。** 追従カメラの記録位置幅は約1.39m。
- 根拠：`evidence/P104_VERDICT.json`、`evidence/P105_CAPTURE_VERDICT.json`、`evidence/case_reports/`、`evidence/p105_camera_audit.json`。
- これは**既知の画像を含む開発条件での実証**であり、未知の実りんごの分類精度や実機の安全性を実証したわけではない。

## 写真・テクスチャを含む完全な実験入力

旧 `g105.zip` では除外していた原本 `p92c(1).zip` を、このP106では**バイトを変更せず3分割して収録**した。GitHubのWebアップロード制限（1ファイル25MB）を避けるため、各ファイルは18MB以下である。

- `assets/p92c_verified.zip.part01`～`part03`: 同一実験原本（50,324,332 bytes）
- 原本SHA-256：`c5927d8122b32a037544a3e3eb0da35f647a6231b3f0cb4eefba2f62cf819928`
- 実験原本の内部に**ユーザー提供りんご写真17枚** (`REAL17_USER_SOURCE_17.zip`)、極部合成UV11枚、等級別のテクスチャ、既存学習用カメラ画像などが含まれる。
- 再結合後の原本ZIPは `.gitignore` で追跡対象から除く。分割ファイル3個はGitHubへ登録する。

## 個別実行（手動）による再現手順

GitHubのPublicリポジトリを `git clone`、またはZIPとしてダウンロードして展開する。**以下のコマンドはGoogle Colab/Linuxを想定**する。Colabの作業ディレクトリはリポジトリ直下へ移動しておくこと。Windowsでは作業パスと日本語フォントの設定が異なる。

```bash
# 1. 写真と実験テクスチャを含む元のZIPを復元（ZIP原本のSHA照合込み）
python reproduction/assemble_assets.py

# 2. 原本、写真17枚、合成UV11枚、等級別テクスチャ4枚を静的に検証
python verify_assets.py

# 3. 公開コードと保存済みP104/P105実験結果を静的に検証
python verify_evidence.py

# 4. 物理シミュレーションP104の全10ケースを新しい作業ディレクトリで実行
python reproduction/p104_run.py --p92c assets/p92c_verified.zip --work /content/P104_REPRO_NEW

# 5. P105のPyBullet追従・把持カメラを新しい別ディレクトリで撮影
python reproduction/p105_camera.py --p92c assets/p92c_verified.zip --work /content/P105_REPRO_NEW
```

環境：成功記録はPython 3.13.18、PyBullet 3.2.7、scikit-learn 1.8.0、trimesh 5.2.0、FFmpeg、Noto CJK。`requirements.txt` に依存関係を記載。Google Colabでは日本語フォント `fonts-noto-cjk` が必要。PyBullet 3.2.7はPython 3.13でソースからビルドされる場合があり、インストール時間・互換性は環境に依存する。既存ランナーは依存導入と事前検査も実行する。

必ず**新規の作業フォルダ**を使用すること。既存の `source` がある場合、P105は二重実行防止のため `STOP: WORKSPACE_NOT_NEW` で停止する。成功済みの元データや結果フォルダに上書きしないこと。

上記1～3の静的監査と素材復元についてはP106作成時に検証済み。P106パッケージ作成時点では**新たな環境での物理試行が未実施**だったが、P107のColab実行で10/10・追加撮影を検証済みである。**任意の第三者環境で必ず成功することを保証するものではない。** 物理実行後は新しい監査JSONで判定し、過去の10/10成功と区別する。

## 提出用完成動画の再現

P104/P105の物理再現とは別に、**提出時に見せる完成動画の再現**に必要な資料を `submission_video/` に追加した。

- `submission_video/p98r2_background.png`：工程98R2で承認済みの固定背景
- `submission_video/p105_compose.py`：P105で得た3本の実録動画と判定JSONから、提出用動画 `PAI_final_p105.mp4` を生成するスクリプト
- `submission_video/p105_layout_bgfix.png`：背景修正版のレイアウト確認画像

実験結果だけを再現するなら不要だが、**第三者が提出用動画まで同じ形で再現するには必要**である。

```bash
python submission_video/p105_compose.py \
  --physics /path/to/stage69_one_world_continuous_1x.mp4 \
  --grip /path/to/p105_grip_live.mp4 \
  --follow /path/to/p105_slider_follow_live.mp4 \
  --verdict /path/to/P105_CAPTURE_VERDICT.json \
  --photo submission_video/p98r2_background.png \
  --out PAI_final_p105.mp4
```

この合成は、**P105の新規実行で得た動画のみ**を使う。旧 `PAI_final_video.mp4` をつないで代用しない。

## ディレクトリの役割

- `source/`: P92C由来の実装コードと成功時の `stage84_camera_ai.py` / `stage90_wound_evidence.py`。原本ZIPからP104実行時に再生成される。
- `reproduction/`: P104・P105の実行ランナー、元アーカイブ復元スクリプト、P105撮影用Colab。
- `assets/`: 元画像・UV・実行データを含む原本のバイト分割ファイル。
- `evidence/`: P104 10件のケース別ログ、P105カメラ監査、SHA照合記録。
- `legacy_stage27_code/`: 設計の履歴参照用。P104で合格した最新実行元とは異なる。
- `verify_evidence.py` / `verify_assets.py`: SHA・保存済み監査の確認のみ。PyBulletの新規実行ではない。

## 公開・権利・制限

このP106には**りんご実写写真17枚と派生画像が含まれる**。GitHubでPublicにする前に、写真とソースコードの公開権限や秘密情報を利用者（アップロード者）が確認すること。パッケージ作成者が勝手にライセンスを付与することはない（`LICENSE_NOT_SPECIFIED.md`参照）。実行証拠が示すことと未検証の内容を区別する。

作業者の手の侵入停止は過去の別工程に模擬検証があるが、P104の全10件に含まれると断定しない。C-2の2本傷の独立視認・一般化精度、現地での実機性能、全ケースでの3秒以下は未実証。

## 提出状況

GitHub Publicへの公開、YouTubeへのアップロード、OmniCampusへの提出は本パッケージ作成だけでは完了しない。講義資料上の締切は2026年11月2日10:00 JST。

## 工程108｜一括ColabのZIP入口

配布ZIP `g108_ready.zip` の `RUN_ALL_COLAB.ipynb` は、`g108_ready.zip` または元の `g108.zip` をアップロード対象として受け付ける。P104/P105の本実行はユーザーのP107で検証済み。新しく実行する場合は空のColabランタイムを使う。背景と動画合成スクリプトは `submission_video/` に入っているが、ノートブックで自動生成されるのはP105の実録素材まで。完成動画を合成するには `submission_video/README.md` を参照する。

## 公開・提出版の利用方法（2026/11/01 04:54:20 JST）

- まず `RUN_ALL_COLAB.ipynb` を開く。**`REPO_URL`** はGitHubに公開済みのリポジトリURLである。未公開でZIPを用いる場合は空欄にし、`g108_public.zip` をアップロードする。
- 実行入口の識別子 `P107_...` は成功済みP107実験の名称を継承している。
- このノートブックはP104・P105の物理実験と監査を行う。**工程98R2の背景を合成して完成動画にする処理は別工程**で、`submission_video/README.md` を参照する。
- `submission_video/` の背景画像とレイアウト画像は公開対象にするため、同フォルダ内 `.gitignore` でルートの画像除外規則を上書きしている。
- 動画・PDF・GitHubの公開とOmniCampusへの提出は、配布ZIPを作っただけでは完了しない。
