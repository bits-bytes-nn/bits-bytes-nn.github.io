// Site behaviors: theme toggle, code-copy, mobile menu, nav highlight, smooth
// scroll, sticky-nav class, share popups, image zoom (GLightbox), tooltips
// (Tippy), and the post table of contents. Vanilla JS, no jQuery.
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
    }
    syncIcon();
    themeToggle.addEventListener('click', function () {
      var next = isDark() ? 'light' : 'dark';
      document.documentElement.setAttribute('data-theme', next);
      try { localStorage.setItem('theme', next); } catch (e) {}
      syncIcon();
    });
  }

  // Copy button on code blocks
  document.querySelectorAll('.post-content .highlight').forEach(function (block) {
    var pre = block.querySelector('pre');
    if (!pre) return;
    var btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'code-copy';
    btn.textContent = 'Copy';
    btn.setAttribute('aria-label', 'Copy code to clipboard');
    block.style.position = 'relative';
    block.appendChild(btn);
    btn.addEventListener('click', function () {
      var code = pre.innerText;
      var done = function (ok) {
        btn.textContent = ok ? 'Copied' : 'Failed';
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

  // Smooth scroll for in-page anchors, offset for the fixed header
  document.querySelectorAll('a[href^="#"]').forEach(function (a) {
    a.addEventListener('click', function (e) {
      var id = a.getAttribute('href');
      if (id.length < 2) return;
      var target = document.querySelector(id);
      if (!target) return;
      e.preventDefault();
      var top = target.getBoundingClientRect().top + window.pageYOffset - 80;
      window.scrollTo({ top: top, behavior: 'smooth' });
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
      window.open(a.getAttribute('href'), 'Share', 'width=550,height=255');
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
    if (img.alt) a.setAttribute('data-glightbox', 'title: ' + img.alt);
    img.parentNode.insertBefore(a, img);
    a.appendChild(img);
  });

  // Image zoom (GLightbox reads .glightbox elements)
  if (window.GLightbox) GLightbox({ selector: '.glightbox' });

  // Tooltips via Tippy.js
  if (Array.isArray(window.tooltips)) {
    window.tooltips.forEach(function (t) { tippy(t[0], t[1]); });
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
