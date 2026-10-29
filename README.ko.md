<div align="center">

# Bits, Bytes and Neural Networks

**AI/ML 논문 리뷰·요약과 기술 가이드를 블로그로 발행합니다.**

[bits-bytes-nn.github.io](https://bits-bytes-nn.github.io) · GitHub Pages에서 서빙하는 Jekyll 정적 사이트

[![Deploy](https://github.com/bits-bytes-nn/bits-bytes-nn.github.io/actions/workflows/jekyll.yml/badge.svg)](https://github.com/bits-bytes-nn/bits-bytes-nn.github.io/actions/workflows/jekyll.yml)
![Ruby](https://img.shields.io/badge/ruby-3.3%2B-red)
![Jekyll](https://img.shields.io/badge/SSG-Jekyll-CC0000)
![GitHub Pages](https://img.shields.io/badge/hosting-GitHub%20Pages-222)

[English README](./README.md)

![Bits, Bytes and Neural Networks](assets/my_header_image_full.jpg)

</div>

## 무엇이 있나

글은 한국어가 기본이고 일부는 영어 번역본이 있습니다. 섹션은 Paper Reviews, Paper
Summaries, Paper Surveys, Tech Guides, Insights 다섯 개입니다. 논문 글·서베이·기술 가이드는
[scholar-lens](https://github.com/bits-bytes-nn/scholar-lens)가 PR로 올리고, Insights는 직접
씁니다. `main`으로 가는 모든 PR에서 전체 검사가 돌고, 머지하면 배포됩니다.

## 빠른 시작

1. **Ruby 3.3 이상과 Bundler.** 저장소의 `.ruby-version`에 버전이 있어 rbenv나 asdf가
   알아서 맞춥니다. macOS 기본 Ruby는 너무 오래됐습니다.
2. **의존성 설치:** `bundle install`
3. **개발 서버**(저장하면 다시 빌드, `http://localhost:4000`):
   `bundle exec jekyll serve`. `_config.yml`을 바꾸면 다시 시작합니다.
4. **CI와 같은 검사:**

   ```bash
   ruby test/run_all.rb && ruby script/lint-posts.rb
   bundle exec jekyll build --strict-front-matter --destination /tmp/site-verify
   bundle exec htmlproofer /tmp/site-verify --disable-external --allow-hash-href --no-ignore-empty-alt
   script/validate-site.sh /tmp/site-verify
   ```

## 문서

상세 문서는 영어로 한 벌만 둡니다.

| 문서 | 내용 |
|---|---|
| [docs/writing.md](docs/writing.md) | 글 작성: 프런트매터, 날짜, 카테고리와 표지, 태그, 번역본, 수식, 이미지, 린트 |
| [docs/operations.md](docs/operations.md) | 검사, CI와 배포, 배포 실패 복구, 게시 URL 기록, scholar-lens, Search Console, 문제 해결 |
| [docs/architecture.md](docs/architecture.md) | 빌드 구조: 파이프라인, 플러그인, URL과 목록, head와 피드, 검색, 스타일, 수식, 검증, 알려진 한계 |

## 라이선스

코드는 MIT, 글(포스트와 그 다이어그램)은 CC BY 4.0입니다. 다른 사람이 만든 그림과
이미지는 원래 라이선스를 따르며, 쓰인 자리에 출처를 밝혀 두었습니다. 제임스 웹 표지 이미지(ESA/Webb,
CC BY 4.0)의 출처를 포함한 자세한 내용은 [LICENSE](LICENSE)에 있습니다.
[Centrarium](https://github.com/bencentra/centrarium) Jekyll 테마를 기반으로 합니다.
