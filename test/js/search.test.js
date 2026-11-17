// Unit tests for js/search.js's matching, ranking and markup, in node:
//   node --test test/js/
// The cases are the ones a review found broken: a title match ranked below
// passing mentions, a translation pair taking two slots, markup in a result.
'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const { prepare, rank, excerpt, mark, resultHtml } = require('../../js/search.js');

const post = (title, extra = {}) => ({ title, url: '/' + title.toLowerCase().replace(/\W+/g, '-') + '.html',
  date: 'January 1, 2026', lang: 'ko', translation: null, category: 'Paper Reviews, Language Models',
  tags: '', content: '', ...extra });

const titles = (results) => results.map((r) => r.doc.post.title);

test('a title match outranks newer posts that only mention the word', () => {
  const docs = prepare([
    post('DSPy', { content: 'compared with Llama once' }),
    post('ReWOO', { content: 'llama llama' }),
    post('The Llama 3 Herd of Models', { content: 'Llama 3 is …' }),
  ]);
  assert.deepEqual(titles(rank(docs, ['llama'], 'ko')).slice(0, 1), ['The Llama 3 Herd of Models']);
});

test('every word must be in the post, in any field', () => {
  const docs = prepare([post('Mixtral', { tags: 'Mixture of Experts', content: '라우팅' }), post('Other', { content: '라우팅' })]);
  assert.deepEqual(titles(rank(docs, ['experts', '라우팅'], 'ko')), ['Mixtral']);
  assert.deepEqual(rank(docs, ['absent'], 'ko'), []);
});

test('ties go to the newer post (index order)', () => {
  const docs = prepare([post('Newer', { content: 'rag' }), post('Older', { content: 'rag' })]);
  assert.deepEqual(titles(rank(docs, ['rag'], 'ko')), ['Newer', 'Older']);
});

test('a translation pair counts once, as the post in the site language', () => {
  const docs = prepare([
    post('데이터에 맥락을', { translation: 'ai-ready', content: 'semantic layer' }),
    post('Writing Context', { lang: 'en', translation: 'ai-ready', content: 'semantic layer semantic layer' }),
  ]);
  const results = rank(docs, ['semantic', 'layer'], 'ko');
  assert.deepEqual(titles(results), ['데이터에 맥락을']);
  // At the better score of the two (the twin's here), so the pair is not ranked down.
  const twinAlone = rank(prepare([docs[1].post]), ['semantic', 'layer'], 'ko')[0].score;
  assert.equal(results[0].score, twinAlone);
});

test('the site-language post wins its pair whichever comes first in the index', () => {
  const docs = prepare([
    post('English twin', { lang: 'en', translation: 'p', content: 'agent' }),
    post('한국어 원문', { translation: 'p', content: 'agent' }),
  ]);
  assert.deepEqual(titles(rank(docs, ['agent'], 'ko')), ['한국어 원문']);
});

test('the twin shows when only it matches', () => {
  const docs = prepare([
    post('한국어 원문', { translation: 'p' }),
    post('English twin', { lang: 'en', translation: 'p', content: 'harness' }),
  ]);
  assert.deepEqual(titles(rank(docs, ['harness'], 'ko')), ['English twin']);
});

test('the date is not searched', () => {
  assert.deepEqual(rank(prepare([post('A', { date: 'October 7, 2025' })]), ['2025'], 'ko'), []);
});

test('marks are escaped, merged, and the only tags added', () => {
  assert.equal(mark('<think> & gpt-4', ['<think>', 'gpt', 'gpt-4']), '<mark>&lt;think&gt;</mark> &amp; <mark>gpt-4</mark>');
  const html = resultHtml(post('<img src=x onerror=alert(1)>', { url: '/a".html', lang: 'ko"' }), ['img']);
  assert.ok(!/<img/.test(html));
  assert.ok(html.includes('href="/a&quot;.html"') && html.includes('lang="ko&quot;"'));
});

test('the excerpt is cut around the match, not at the top', () => {
  const text = 'x '.repeat(400) + 'the needle is here ' + 'y '.repeat(400);
  const cut = excerpt(text, ['needle']);
  assert.ok(cut.startsWith('…') && cut.endsWith('…'));
  assert.ok(cut.includes('<mark>needle</mark>'));
});
