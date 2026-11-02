# Original assets for P104/P105 reproduction

作成：2026/10/31 18:21:50 JST

- 実験原本 `p92c(1).zip`（50,324,332 bytes、SHA-256 `c5927d8122b32a037544a3e3eb0da35f647a6231b3f0cb4eefba2f62cf819928`）をバイト改変なしで3分割した。
- 17枚のユーザー提供写真、11枚の極部合成UV、実行用の画像テクスチャが含まれる。
- GitHubのWebアップロード単体ファイル25MB制限を考慮して1パート18,000,000バイト以下にしている。
- 再結合：`python reproduction/assemble_assets.py`。原本ZIPは`.gitignore`によりGitの追跡対象にならない。

| part | bytes | SHA-256 |
|---|---:|---|
| p92c_verified.zip.part01 | 18000000 | `728263247899de256962fd696e70dada23649f61c0fe8dee50278bedd94f0a57` |
| p92c_verified.zip.part02 | 18000000 | `d1c51364bdbf75e9f35034f947699a1e5048b3dd51cef5744ce14a155c3d0ce7` |
| p92c_verified.zip.part03 | 14324332 | `306e0a9fc0f6bdfca284b37b570f37e17ad1c8e6fcd228534a735e1372eb5b0f` |
