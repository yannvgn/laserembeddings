<a name="1.1.2"></a>
# [1.1.2](https://github.com/yannvgn/laserembeddings/compare/v1.1.1...v1.1.2) (2021-12-12)

- A compatibility issue with subword-nmt 0.3.8 was fixed (#39) 🐛
- The behavior of `Laser.embed_sentences` was unclear/misleading when the number of language codes received in the `lang` argument did not match the number of sentences to encode. It now raises an error in that case (#40) 🐛

<a name="1.1.1"></a>
# [1.1.1](https://github.com/yannvgn/laserembeddings/compare/v1.1.0...v1.1.1) (2021-02-06)

- An issue with PyTorch 1.7.0 was fixed (#32) 🐛

<a name="1.1.0"></a>
# [1.1.0](https://github.com/yannvgn/laserembeddings/compare/v1.0.1...v1.1.0) (2020-10-04)

- Japanese extra on Windows is back! 🇯🇵

<a name="1.0.1"></a>
# [1.0.1](https://github.com/yannvgn/laserembeddings/compare/v1.0.0...v1.0.1) (2020-03-02)

- The encoder was fixed to remove an innocuous warning message that would sometimes appear when using PyTorch 1.4 🐛
- Japanese extra is now disabled on Windows (sorry) to prevent installation issues and computation failures in other languages 😕

<a name="1.0.0"></a>
# [1.0.0](https://github.com/yannvgn/laserembeddings/compare/v0.1.3...v1.0.0) (2019-12-19)

- Greek, Chinese and Japanese are now supported 🇬🇷 🇨🇳 🇯🇵 
- Some languages that were only partially supported are now fully supported (New Norwegian, Swedish, Tatar) 🌍
- It should work on Windows now 🙄
- Sentences in different languages can now be processed in the same batch ⚡️

<a name="0.1.3"></a>
# [0.1.3](https://github.com/yannvgn/laserembeddings/compare/v0.1.2...v0.1.3) (2019-10-03)

- A lot of languages that were only partially supported are now fully supported (br, bs, ceb, fr, gl, oc, ug, vi) 🌍

<a name="0.1.2"></a>
# [0.1.2](https://github.com/yannvgn/laserembeddings/compare/v0.1.1...v0.1.2) (2019-08-24)

- Korean is now fully supported ✅
- A [bug](https://bugs.python.org/issue37723) in Python 3.7 (<= 3.7.4) and 3.8 (<= 3.8.0 beta 3) affecting the tokenizer performance was patched as a temporary solution until next Python releases 🐛

<a name="0.1.1"></a>
# 0.1.1 (2019-07-23)

- Initial version 🐣
