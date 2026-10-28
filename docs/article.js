'use strict';
const conditions = {
  '4': { baseline: 34, candidate: 42, counts: [17, 21], note: 'Original appearance: paired p = 0.4545. Material improvement remains inconclusive.' },
  '0': { baseline: 12, candidate: 28, counts: [6, 14], note: 'Anti-aliasing disabled: paired p = 0.0574. This comparison also remains inconclusive; rendering changes observed outcomes.' }
};
document.querySelectorAll('[data-render]').forEach(button => button.addEventListener('click', () => {
  const c = conditions[button.dataset.render];
  document.querySelectorAll('[data-render]').forEach(b => { b.classList.toggle('active', b === button); b.setAttribute('aria-pressed', String(b === button)); });
  document.getElementById('upstream-rate').textContent = `${c.baseline}%`;
  document.getElementById('candidate-rate').textContent = `${c.candidate}%`;
  document.getElementById('upstream-count').textContent = `${c.counts[0]} / 50 strict placements`;
  document.getElementById('candidate-count').textContent = `${c.counts[1]} / 50 strict placements`;
  document.getElementById('render-note').textContent = c.note;
}));
if ('IntersectionObserver' in window) {
  const links = Array.from(document.querySelectorAll('.contents nav a'));
  const observer = new IntersectionObserver(entries => entries.forEach(entry => {
    if (!entry.isIntersecting) return;
    links.forEach(link => link.classList.toggle('current', link.hash === `#${entry.target.id}`));
  }), { rootMargin: '-10% 0px -65% 0px', threshold: 0 });
  document.querySelectorAll('.story section').forEach(section => observer.observe(section));
}
