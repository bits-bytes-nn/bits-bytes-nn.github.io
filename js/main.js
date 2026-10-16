// Site behaviors: theme toggle, code-copy, mobile menu, nav highlight,
// sticky-nav class, image zoom (GLightbox), and the post table of contents.
// Vanilla JS, no jQuery. The fixed-header offset for in-page jumps is CSS
// (`scroll-padding-top` in _sass/_layout.scss); this keeps its --nav-height
// in step with the bar.
document.addEventListener('DOMContentLoaded', function () {
  // Dark-mode toggle. Light is the default; dark is opt-in and persisted.
  // The OS setting is intentionally NOT followed.
  var themeToggle = document.getElementById('theme-toggle');
  if (themeToggle) {
    var themeColor = document.querySelector('meta[name="theme-color"]');
    function isDark() {
      return document.documentElement.getAttribute('data-theme') === 'dark';
    }
    function syncTheme() {
      // aria-label is static, so without this a screen-reader user cannot tell
      // whether dark mode is currently on.
      themeToggle.setAttribute('aria-pressed', isDark() ? 'true' : 'false');
      // Repaints the mobile browser's own chrome. The inline bootstrap in
      // head.html does this on load; without it here the address bar stayed
      // white until the next navigation. Keep in step with $bg in _dark.scss.
      if (themeColor) themeColor.setAttribute('content', isDark() ? '#0d1117' : '#ffffff');
    }
    syncTheme();
    themeToggle.addEventListener('click', function () {
      var next = isDark() ? 'light' : 'dark';
      document.documentElement.setAttribute('data-theme', next);
      try { localStorage.setItem('theme', next); } catch (e) {}
      syncTheme();
    });
  }

  // Copy button on code blocks.
  //
  // Deliberately no aria-label: per accname, aria-label wins over element
  // contents, so a static "Copy code to clipboard" froze the accessible name and
  // the "Copied"/"Failed" result — the only feedback this control exists to give —
  // was never announced. The visible text is the name, and one shared live region
  // reports the outcome. It is shared because a long post has ~50 of these.
  var copyStatus = null;
  function announceCopy(msg) {
    if (!copyStatus) {
      copyStatus = document.createElement('span');
      copyStatus.className = 'sr-only';
      copyStatus.setAttribute('aria-live', 'polite');
      document.body.appendChild(copyStatus);
    }
    copyStatus.textContent = msg;
  }
  document.querySelectorAll('.post-content .highlight').forEach(function (block) {
    var pre = block.querySelector('pre');
    if (!pre) return;
    var btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'code-copy';
    btn.textContent = 'Copy';
    block.style.position = 'relative';
    block.appendChild(btn);
    btn.addEventListener('click', function () {
      var code = pre.innerText;
      var done = function (ok) {
        btn.textContent = ok ? 'Copied' : 'Failed';
        announceCopy(ok ? 'Copied to clipboard' : 'Copy failed');
        setTimeout(function () { btn.textContent = 'Copy'; }, 1500);
      };
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(code).then(function () { done(true); }, function () { done(false); });
      } else {
        // Fallback for non-secure contexts / older browsers
        try {
          var ta = document.createElement('textarea');
          ta.value = code; ta.style.position = 'fixed'; ta.style.opacity = '0';
          document.body.appendChild(ta); ta.select();
          done(document.execCommand('copy'));
          document.body.removeChild(ta);
        } catch (e) { done(false); }
      }
    });
  });

  // Mobile menu toggle
  var menuToggle = document.getElementById('js-mobile-menu');
  var menu = document.getElementById('js-navigation-menu');
  if (menu) menu.classList.remove('show');
  if (menuToggle && menu) {
    menuToggle.addEventListener('click', function (e) {
      e.preventDefault();
      var open = menu.classList.toggle('show');
      menuToggle.setAttribute('aria-expanded', open ? 'true' : 'false');
    });
    // Escape is the expected way out of an expanded disclosure, whether focus is
    // in the menu or still on the button that opened it.
    function closeOnEscape(e) {
      if (e.key !== 'Escape' || !menu.classList.contains('show')) return;
      menu.classList.remove('show');
      menuToggle.setAttribute('aria-expanded', 'false');
      menuToggle.focus();
    }
    menu.addEventListener('keydown', closeOnEscape);
    menuToggle.addEventListener('keydown', closeOnEscape);
    // Tabbing out of the open menu would otherwise leave it covering the page
    // the focus has moved into.
    menu.addEventListener('focusout', function (e) {
      var to = e.relatedTarget; // null when focus leaves the window, not the menu
      if (!menu.classList.contains('show') || !to || menu.contains(to) || to === menuToggle) return;
      menu.classList.remove('show');
      menuToggle.setAttribute('aria-expanded', 'false');
    });
  }

  // Highlight the current page in the nav
  var here = window.location.pathname;
  document.querySelectorAll('.nav-link a').forEach(function (link) {
    var path = new URL(link.getAttribute('href'), window.location.origin).pathname;
    // A post sits under its section's path (/paper-reviews/…): there the link is
    // where the reader is, not the page itself.
    if (here === path || (path !== '/' && here.indexOf(path) === 0)) {
      link.classList.add('active');
      link.setAttribute('aria-current', here === path ? 'page' : 'location');
    }
  });

  // Add a class to the nav once the page is scrolled
  var nav = document.querySelector('.navigation');
  if (nav) {
    // The page starts below the fixed bar, which grows to two rows when its
    // tabs do not fit on one. The phone menu's dropdown is not the bar: there
    // the bar stays one row and the open menu covers the page on purpose.
    var wide = window.matchMedia('(min-width: 768px)'); // $tablet, as in default.html
    var syncNavHeight = function () {
      if (wide.matches) document.documentElement.style.setProperty('--nav-height', nav.offsetHeight + 'px');
      else document.documentElement.style.removeProperty('--nav-height');
    };
    if (window.ResizeObserver) new ResizeObserver(syncNavHeight).observe(nav);
    else window.addEventListener('resize', syncNavHeight);
    if (wide.addEventListener) wide.addEventListener('change', syncNavHeight);
    else wide.addListener(syncNavHeight); // Safari before 14
    syncNavHeight();
    window.addEventListener('scroll', function () {
      nav.classList.toggle('scrolled', window.pageYOffset > 50);
    }, { passive: true });
  }

  // Wrap plain Markdown images in a .glightbox anchor so they're zoomable too
  // (hand-written <a class="glightbox"> images are already covered).
  document.querySelectorAll('.post-content img').forEach(function (img) {
    if (img.closest('a')) return;                 // already linked (e.g. glightbox)
    if (img.classList.contains('profile')) return; // about-page portrait: not zoomable
    var src = img.getAttribute('src');
    if (!src) return;
    var a = document.createElement('a');
    a.href = src;
    a.className = 'glightbox';
    a.setAttribute('data-gallery', 'post-images');
    // data-title, not data-glightbox: GLightbox parses the latter by splitting on
    // ';' then ':', so alts like "TL;DR Best of N" or "Chain-of-Thought: …"
    // produced a mangled caption.
    if (img.alt) a.setAttribute('data-title', img.alt);
    img.parentNode.insertBefore(a, img);
    a.appendChild(img);
  });

  // Image zoom (GLightbox reads .glightbox elements). On close, focus returns to
  // the image that opened it, so a screen reader keeps its place.
  var english = document.documentElement.lang.indexOf('en') === 0;

  if (window.GLightbox) {
    var opener = null;
    // `focused` is the class GLightbox's own Tab trap looks for; without it
    // Tab leaves the dialog.
    var focusClose = function () {
      var close = document.querySelector('.glightbox-container .gclose');
      if (close) { close.focus(); close.classList.add('focused'); }
    };
    document.addEventListener('click', function (e) {
      var link = e.target.closest && e.target.closest('.glightbox');
      if (link) opener = link; // not the lightbox's own close button
    }, true);
    // Below 769px GLightbox moves the arrows off-screen (swipe instead) but its
    // Tab trap still cycles them; take them out of it there, put them back above.
    var narrow = window.matchMedia('(max-width: 768px)');
    var syncArrows = function () {
      document.querySelectorAll('.glightbox-container .gprev, .glightbox-container .gnext').forEach(function (btn) {
        if (!btn.hasAttribute('data-order')) btn.setAttribute('data-order', btn.getAttribute('data-taborder') || '');
        if (narrow.matches) {
          btn.removeAttribute('data-taborder');
          btn.setAttribute('tabindex', '-1');
          if (document.activeElement === btn) { // it just went off-screen
            btn.classList.remove('focused');
            focusClose();
          }
        } else {
          btn.setAttribute('data-taborder', btn.getAttribute('data-order'));
          btn.removeAttribute('tabindex');
        }
      });
    };
    var lightbox = GLightbox({ selector: '.glightbox' }); // .on() does not chain
    lightbox.on('open', function () {
      // A dialog: focus moves into it, and the page behind is inert to readers.
      var box = document.querySelector('.glightbox-container');
      if (box) {
        box.setAttribute('aria-modal', 'true');
        box.setAttribute('aria-label', english ? 'Image viewer' : '이미지 보기');
      }
      focusClose();
      syncArrows();
      narrow.addEventListener('change', syncArrows); // a tablet turning
    });
    lightbox.on('close', function () {
      narrow.removeEventListener('change', syncArrows);
      if (opener) opener.focus();
    });
  }

  // Spoken, never shown, so in the page's language: a Korean voice reading
  // "Table, scrollable" is unintelligible. (The visible chrome stays English.)
  // Numbered, so a reader's landmark list tells one table from the next.
  function scrollerLabel(el) {
    var code = el.tagName === 'PRE';
    var n = Array.prototype.indexOf.call(document.querySelectorAll(code ? '.post-content pre' : '.post-content .table-container'), el) + 1;
    if (code) return english ? 'Code block ' + n + ', scrollable' : '코드 블록 ' + n + ', 가로 스크롤';
    return english ? 'Table ' + n + ', scrollable' : '표 ' + n + ', 가로 스크롤';
  }

  // A horizontally scrolling box can only be scrolled by keyboard if it is
  // focusable — Safari and Firefox have no equivalent of Chrome's keyboard-
  // focusable scrollers. This is done here rather than in the templates because
  // it depends on measurement: only containers that actually overflow become tab
  // stops, so tables that fit add nothing to the tab order.
  // Re-measured on resize, since rotating a phone changes which ones overflow.
  var scrollers = document.querySelectorAll('.post-content .table-container, .post-content pre');
  function markScrollers() {
    scrollers.forEach(function (el) {
      var overflows = el.scrollWidth > el.clientWidth + 1;
      if (overflows && !el.hasAttribute('tabindex')) {
        el.setAttribute('tabindex', '0');
        el.setAttribute('role', 'region');
        el.setAttribute('aria-label', scrollerLabel(el));
      } else if (!overflows && el.getAttribute('tabindex') === '0') {
        el.removeAttribute('tabindex');
        el.removeAttribute('role');
        el.removeAttribute('aria-label');
      }
    });
  }
  if (scrollers.length) {
    markScrollers();
    // Lazy images and MathJax can widen a table after DOMContentLoaded.
    window.addEventListener('load', markScrollers);
    var resizeTimer;
    window.addEventListener('resize', function () {
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(markScrollers, 150);
    }, { passive: true });
  }

  // Build a table of contents from post h2 headings (3+ only). The <details>
  // ships with `open`, so it starts expanded; visitors can collapse it.
  var toc = document.getElementById('post-toc');
  if (toc) {
    var heads = document.querySelectorAll('.post-content h2[id]');
    if (heads.length >= 3) {
      var ul = toc.querySelector('ul');
      // Headings that number themselves ("1. 배경") would read "2 1. 배경". Most of
      // them, not one: a lone "1. …" among plain headings keeps the numbers.
      var numbered = Array.prototype.filter.call(heads, function (h) { return /^\s*\d+[.)]\s/.test(h.textContent); }).length > heads.length / 2;
      heads.forEach(function (h, i) {
        var li = document.createElement('li');
        var a = document.createElement('a');
        var num = document.createElement('span');
        a.href = '#' + h.id;
        num.className = 'toc-num';
        num.textContent = i + 1;
        // Text node, not innerHTML: a heading that names a tag ("<think> 토큰")
        // would otherwise be parsed as markup.
        if (numbered) a.append(h.textContent);
        else a.append(num, h.textContent);
        li.appendChild(a);
        ul.appendChild(li);
      });
      var count = toc.querySelector('.post-toc-count');
      if (count) count.textContent = heads.length;
      toc.hidden = false;
    }
  }
});
