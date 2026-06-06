# Bits, Bytes and Neural Networks

> AI/ML 논문을 깊이 있게 뜯어본 리뷰와 최신 연구 노트를 담은 블로그. Jekyll로 만들어 GitHub Pages에 배포합니다.

![Bits, Bytes and Neural Networks](assets/my_header_image_full.jpg)

---

## 빠른 시작

Jekyll 사이트를 처음 다뤄도 따라올 수 있도록 전체 흐름을 정리했습니다.

**1. Ruby 3.3 이상과 Bundler를 준비합니다.** 먼저 설치된 버전을 확인하세요.

```bash
ruby --version     # 3.3 이상 필요 (.ruby-version에 명시)
bundle --version   # Ruby에 기본 포함. 없으면: gem install bundler
```

macOS 기본 Ruby는 너무 오래됐습니다. [rbenv](https://github.com/rbenv/rbenv)나 asdf로
3.3 이상을 설치하세요. 저장소에 `.ruby-version`이 있으니 버전 매니저가 알아서 맞춰 줍니다.

**2. 프로젝트 의존성을 설치합니다** (Jekyll, 플러그인, html-proofer).

```bash
bundle install
```

**3. 개발 서버를 띄웁니다.** 파일을 저장할 때마다 자동으로 다시 빌드하고
`http://localhost:4000`에서 보여 줍니다.

```bash
bundle exec jekyll serve
```

`_posts/`, `_sass/`, `_includes/` 아래 파일을 고치고 저장한 뒤 새로고침하면
대부분 바로 반영됩니다. 단, `_config.yml`을 바꿨을 때는 서버를 다시 시작해야 합니다.

**4. 최종 결과물이 필요하면 프로덕션 빌드를 합니다** (CI가 하는 것과 동일).

```bash
bundle exec jekyll build   # 결과물은 _site/ 에 생성
```

> **왜 `github-pages` gem이 아니라 순수 `jekyll`인가요?**
> 이 사이트는 `_plugins/`에 직접 만든 Ruby 플러그인(읽기 시간, 이미지 lazy-load)을 씁니다.
> `github-pages` gem은 보안 샌드박스 때문에 커스텀 플러그인을 막으므로,
> 로컬과 CI 모두 Jekyll을 직접 실행합니다.

---

## 저장소 구조

```
_posts/            글 — YYYY-MM-DD-slug.md (한국어 기본, 영어 번역본은 -en.md)
_layouts/          페이지 템플릿: default → post / page / archive
_includes/         재사용 조각: head, header, footer, nav_links,
                   page_divider, category-posts, language_switcher
_sass/             스타일: _layout, _post, _tags, _syntax(Rouge 코드 테마),
                   _dark(다크모드), base/*
                   ⚠ bourbon/ · neat/ 는 벤더링된 프레임워크 — 수정하지 말 것
_plugins/          reading_time.rb (한·영 읽기 시간 계산)
                   lazy_images.rb  (<img>에 loading="lazy" 추가)
css/               main.scss(Sass 진입점) · search.css(검색 페이지 전용)
js/                main.js(테마 토글·코드 복사·목차·메뉴·이미지 확대 등)
                   search.js(검색창 동작)
assets/images/     토픽별로 재사용하는 공용 커버 이미지
assets/<slug>/     글마다 하나씩 두는 그림 폴더
search.json        전체 본문 검색 색인 (simple-jekyll-search가 사용)
.github/workflows/ CI: 빌드 → html-proofer 링크 검사 → main 푸시 시 배포
```

**최상위 페이지:** `index.html`(홈), 네 개의 섹션 탭인 `paper-reviews.md`·
`paper-summaries.md`·`tech-guides.md`·`insights.md`, 그리고 `categories.html`·
`tags.html`·`search.md`·`about.md`.

---

## 글 작성하기

가장 쉬운 방법은 `/write-post` 스킬입니다. 리서치 → 초안 → 퇴고까지 한 번에 진행해 줍니다.
직접 만들 때는 `_posts/YYYY-MM-DD-slug.md` 파일을 만들고 아래 프런트매터로 시작하세요.

```yaml
---
layout: post
title: "<글 제목>"
date: YYYY-MM-DD HH:MM:SS
author: "<저자>"                   # 논문 저자(기관). Insights·의견 글이면 생략
categories: ["<유형>", "<주제>"]
tags: ["<태그-1>", "<태그-2>"]
cover: /assets/images/<topic>.(jpg|png)
use_math: true                     # 수식이 있을 때만 (MathJax를 불러옵니다)
lang: ko                           # 선택 — 아래 translation_id와 함께 쓰면…
translation_id: <공통-슬러그>       # …한국어 글과 -en 번역본을 연결합니다
---
```

### 카테고리가 URL을 결정합니다

카테고리는 **두 단계**입니다.

- `categories[0]` — **유형**: `Paper Reviews`, `Paper Summaries`, `Tech Guides`,
  `Insights` 중 하나. 글이 어느 내비 탭에 들어갈지를 정합니다.
- `categories[1]` — **주제**: `Language-Models`, `Multimodal-Learning`,
  `Finetuning`, `Retrieval-Augmented-Generation`, `Agentic-AI` 등. 필요하면 자유롭게 추가합니다.

Jekyll은 이 둘과 날짜를 합쳐 출력 경로를 만듭니다.

```
categories: ["Paper Reviews", "Language-Models"] + date: 2025-01-23
        ↓
_site/paper reviews/language-models/2025/01/23/<slug>.html
```

### 수식은 반드시 `$$…$$`로

인라인이든 디스플레이든 `$$…$$`로 쓰고 `use_math: true`를 켜세요.

**단일 `$…$`는 절대 쓰지 마세요.** kramdown은 단일 `$`를 수식으로 보지 않습니다.
그래서 MathJax가 돌기 *전에* 마크다운 단계가 `$` 안의 `_`·`*`를 `<em>`·`<strong>`으로
바꿔 버립니다 — 예를 들어 `$a*b*c$`가 `$a<em>b</em>c$`가 되어 깨집니다. `$$`로 감싸면
kramdown이 내용을 그대로 보존해 `\(…\)`로 내보내므로 안전합니다. (`$10M` 같은 본문 속
달러 표기는 수식이 아니니 그대로 둬도 됩니다.) 자세한 내용은 [tech-doc.md](tech-doc.md).

### 푸시 전에 확인하세요

```bash
bundle exec jekyll build                              # 빌드가 깨끗한가?
bundle exec htmlproofer ./_site --disable-external    # 깨진 링크·이미지는 없나?
```

CI도 같은 html-proofer 검사를 돌립니다. 로컬에서 미리 잡으면 배포 실패를 막을 수 있습니다.

---

## 배포

`main`에 푸시하면 `.github/workflows/jekyll.yml`이 다음을 수행합니다.

1. `JEKYLL_ENV=production`으로 사이트를 빌드하고
2. `_site/`에 **html-proofer**를 돌려 내부 링크·이미지·앵커를 검사한 뒤
3. GitHub Pages에 배포합니다.

워크플로우가 실패한다면 대개 2단계입니다. Actions 로그를 열면 어떤 링크·이미지가
깨졌는지 정확히 나옵니다. 수동 배포 단계는 없습니다.

> **⚠ `google*.html` / `naver*.html`을 `_config.yml`의 `exclude`에 넣지 마세요.**
> Search Console·네이버 소유권 인증 토큰이라 사이트 루트에 그대로 올라가야 합니다.
> 제외하면 검색 색인이 조용히 망가집니다 — 증상은 "구글이 사이트맵을 못 읽음"입니다.

---

## 라이선스

MIT — [LICENSE.md](LICENSE.md) 참고. [Centrarium](http://jekyllthemes.org/themes/centrarium/)
Jekyll 테마를 기반으로 합니다.
