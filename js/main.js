// Site behaviors: theme toggle, code-copy, mobile menu, nav highlight, smooth
// scroll, sticky-nav class, share popups, image zoom (GLightbox), and the post
// table of contents. Vanilla JS, no jQuery.
document.addEventListener('DOMContentLoaded', function () {
  // Dark-mode toggle. Light is the default; dark is opt-in and persisted.
  // The OS setting is intentionally NOT followed.
  var themeToggle = document.getElementById('theme-toggle');
  if (themeToggle) {
    var icon = themeToggle.querySelector('i');
    function isDark() {
      return document.documentElement.getAttribute('data-theme') === 'dark';
    }
    function syncIcon() {
      if (icon) icon.className = isDark() ? 'fa-solid fa-sun' : 'fa-regular fa-moon';
      // aria-label is static, so without this a screen-reader user cannot tell
      // whether dark mode is currently on.
      themeToggle.setAttribute('aria-pressed', isDark() ? 'true' : 'false');
    }
    syncIcon();
    themeToggle.addEventListener('click', function () {
      var next = isDark() ? 'light' : 'dark';
      document.documentElement.setAttribute('data-theme', next);
      try { localStorage.setItem('theme', next); } catch (e) {}
      syncIcon();
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
    // Escape is the expected way out of an expanded disclosure.
    menu.addEventListener('keydown', function (e) {
      if (e.key !== 'Escape' || !menu.classList.contains('show')) return;
      menu.classList.remove('show');
      menuToggle.setAttribute('aria-expanded', 'false');
      menuToggle.focus();
    });
  }

  // Highlight the current page in the nav
  var here = window.location.pathname;
  document.querySelectorAll('.nav-link a').forEach(function (link) {
    var path = new URL(link.getAttribute('href'), window.location.origin).pathname;
    if (here === path) {
      link.classList.add('active');
      link.setAttribute('aria-current', 'page');
    }
  });

  // Smooth scroll for in-page anchors, offset for the fixed header.
  //
  // preventDefault() cancels the native fragment navigation, so focus and the URL
  // hash have to be moved by hand — otherwise a reader who clicks a
  // table-of-contents entry keeps focus on the link, Tab continues from the TOC
  // rather than the section, and a screen reader announces nothing at all.
  //
  // 'smooth' is passed explicitly, which overrides the `scroll-behavior: auto
  // !important` in the reduced-motion block in _sass/_layout.scss, so that
  // preference is checked here too.
  var reduceMotion = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)');
  document.querySelectorAll('a[href^="#"]').forEach(function (a) {
    a.addEventListener('click', function (e) {
      var id = a.getAttribute('href');
      if (id.length < 2) return;
      var target = document.querySelector(id);
      if (!target) return;
      e.preventDefault();
      var top = target.getBoundingClientRect().top + window.pageYOffset - 80;
      window.scrollTo({ top: top, behavior: reduceMotion && reduceMotion.matches ? 'auto' : 'smooth' });
      // -1 so headings stay out of the tab order but can still receive focus.
      if (!target.hasAttribute('tabindex')) target.setAttribute('tabindex', '-1');
      target.focus({ preventScroll: true });
      if (history.replaceState) history.replaceState(null, '', id);
    });
  });

  // Add a class to the nav once the page is scrolled
  var nav = document.querySelector('.navigation');
  if (nav) {
    window.addEventListener('scroll', function () {
      nav.classList.toggle('scrolled', window.pageYOffset > 50);
    }, { passive: true });
  }

  // Share links open in a small popup window
  document.querySelectorAll('.js-share-popup').forEach(function (a) {
    a.addEventListener('click', function (e) {
      e.preventDefault();
      window.open(a.getAttribute('href'), 'Share', 'noopener');
    });
  });

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
    // ';' then ':', so alts like "TL;DR Best of N" or "Chain-of-Thought: …" —
    // 10 of them on this site — produced a mangled caption.
    if (img.alt) a.setAttribute('data-title', img.alt);
    img.parentNode.insertBefore(a, img);
    a.appendChild(img);
  });

  // Image zoom (GLightbox reads .glightbox elements)
  if (window.GLightbox) GLightbox({ selector: '.glightbox' });

  // A horizontally scrolling box can only be scrolled by keyboard if it is
  // focusable — Safari and Firefox have no equivalent of Chrome's keyboard-
  // focusable scrollers. This is done here rather than in the templates because
  // it depends on measurement: only containers that actually overflow become tab
  // stops, so the 200-odd tables that fit add nothing to the tab order.
  // Re-measured on resize, since rotating a phone changes which ones overflow.
  var scrollers = document.querySelectorAll('.post-content .table-container, .post-content pre.highlight');
  function markScrollers() {
    scrollers.forEach(function (el) {
      var overflows = el.scrollWidth > el.clientWidth + 1;
      if (overflows && !el.hasAttribute('tabindex')) {
        el.setAttribute('tabindex', '0');
        el.setAttribute('role', 'region');
        el.setAttribute('aria-label', el.tagName === 'PRE' ? 'Code block, scrollable' : 'Table, scrollable');
      } else if (!overflows && el.getAttribute('tabindex') === '0') {
        el.removeAttribute('tabindex');
        el.removeAttribute('role');
        el.removeAttribute('aria-label');
      }
    });
  }
  if (scrollers.length) {
    markScrollers();
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
      heads.forEach(function (h, i) {
        var li = document.createElement('li');
        var a = document.createElement('a');
        a.href = '#' + h.id;
        a.innerHTML = '<span class="toc-num">' + (i + 1) + '</span>' + h.textContent;
        li.appendChild(a);
        ul.appendChild(li);
      });
      var count = toc.querySelector('.post-toc-count');
      if (count) count.textContent = heads.length;
      toc.hidden = false;
    }
  }
});
