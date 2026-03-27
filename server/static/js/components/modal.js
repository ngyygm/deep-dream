/* ==========================================
   Modal Dialog Component
   ========================================== */

function showModal({ title, content, footer, onClose, size = 'md' }) {
  const overlay = document.createElement('div');
  overlay.className = 'modal-overlay';

  const widthMap = { sm: '400px', md: '600px', lg: '800px', xl: '1000px' };

  overlay.innerHTML = `
    <div class="modal" style="max-width:${widthMap[size] || widthMap.md}">
      <div class="modal-header">
        <h3 style="font-size:1rem;font-weight:600;color:var(--text-primary);margin:0;">${escapeHtml(title)}</h3>
        <button class="btn btn-ghost btn-sm modal-close-btn">
          <i data-lucide="x" style="width:16px;height:16px;"></i>
        </button>
      </div>
      <div class="modal-body">${content}</div>
      ${footer ? `<div class="modal-footer">${footer}</div>` : ''}
    </div>
  `;

  const close = () => {
    overlay.style.animation = 'modal-fade-in 0.15s ease reverse';
    setTimeout(() => {
      overlay.remove();
      if (onClose) onClose();
    }, 150);
  };

  overlay.querySelector('.modal-close-btn').addEventListener('click', close);
  overlay.addEventListener('click', (e) => {
    if (e.target === overlay) close();
  });

  document.body.appendChild(overlay);
  if (window.lucide) lucide.createIcons({ nodes: [overlay] });

  return { overlay, close };
}
