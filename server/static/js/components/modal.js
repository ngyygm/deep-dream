/* ==========================================
   Modal Dialog Component — a11y enhanced
   ========================================== */

let _activeModal = null;

function showModal({ title, content, footer, onClose, size = 'md' }) {
  // Close any existing modal first
  if (_activeModal) _activeModal.close();

  const overlay = document.createElement('div');
  overlay.className = 'modal-overlay';
  overlay.setAttribute('role', 'dialog');
  overlay.setAttribute('aria-modal', 'true');
  overlay.setAttribute('aria-label', title);

  const widthMap = { sm: '400px', md: '600px', lg: '800px', xl: '1000px' };

  overlay.innerHTML = `
    <div class="modal" style="max-width:${widthMap[size] || widthMap.md}" tabindex="-1">
      <div class="modal-header">
        <h3 style="font-size:1rem;font-weight:600;color:var(--text-primary);margin:0;">${escapeHtml(title)}</h3>
        <button class="btn btn-ghost btn-sm modal-close-btn" aria-label="Close dialog">
          <i data-lucide="x" style="width:16px;height:16px;"></i>
        </button>
      </div>
      <div class="modal-body">${content}</div>
      ${footer ? `<div class="modal-footer">${footer}</div>` : ''}
    </div>
  `;

  const modal = overlay.querySelector('.modal');
  let _previousFocus = document.activeElement;

  const close = () => {
    overlay.style.animation = 'modal-fade-in 0.15s ease reverse';
    setTimeout(() => {
      overlay.remove();
      _activeModal = null;
      if (_previousFocus) _previousFocus.focus();
      if (onClose) onClose();
    }, 150);
  };

  // Close button
  overlay.querySelector('.modal-close-btn').addEventListener('click', close);

  // Click outside to close
  overlay.addEventListener('click', (e) => {
    if (e.target === overlay) close();
  });

  // Focus trap
  const _focusTrap = (e) => {
    if (e.key !== 'Tab') return;
    const focusable = modal.querySelectorAll(
      'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
    );
    if (focusable.length === 0) return;
    const first = focusable[0];
    const last = focusable[focusable.length - 1];
    if (e.shiftKey) {
      if (document.activeElement === first) {
        e.preventDefault();
        last.focus();
      }
    } else {
      if (document.activeElement === last) {
        e.preventDefault();
        first.focus();
      }
    }
  };

  // Escape key
  const _onKeyDown = (e) => {
    if (e.key === 'Escape') {
      e.stopPropagation();
      close();
    }
  };

  overlay.addEventListener('keydown', _focusTrap);
  overlay.addEventListener('keydown', _onKeyDown);

  document.body.appendChild(overlay);
  if (window.lucide) lucide.createIcons({ nodes: [overlay] });

  // Auto-focus first focusable element
  requestAnimationFrame(() => {
    const first = modal.querySelector(
      'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
    );
    if (first) first.focus();
    else modal.focus();
  });

  const handle = { overlay, close };
  _activeModal = handle;
  return handle;
}

/**
 * Styled confirmation dialog — replaces window.confirm().
 * Returns a Promise<boolean>.
 */
function showConfirm({ title, message, confirmLabel, cancelLabel, destructive }) {
  return new Promise((resolve) => {
    const confirmBtnClass = destructive ? 'btn btn-danger' : 'btn btn-primary';
    const footer = `
      <div style="display:flex;justify-content:flex-end;gap:0.5rem;">
        <button class="btn btn-secondary confirm-cancel-btn">${escapeHtml(cancelLabel || t('common.cancel') || 'Cancel')}</button>
        <button class="${confirmBtnClass} confirm-ok-btn">${escapeHtml(confirmLabel || t('common.confirm') || 'OK')}</button>
      </div>
    `;
    const modal = showModal({
      title: title || t('common.confirm') || 'Confirm',
      content: `<p style="font-size:0.875rem;color:var(--text-secondary);margin:0;line-height:1.6;">${escapeHtml(message)}</p>`,
      footer,
      onClose: () => resolve(false),
    });
    const okBtn = modal.overlay.querySelector('.confirm-ok-btn');
    const cancelBtn = modal.overlay.querySelector('.confirm-cancel-btn');
    if (okBtn) okBtn.addEventListener('click', () => { modal.close(); resolve(true); });
    if (cancelBtn) cancelBtn.addEventListener('click', () => { modal.close(); resolve(false); });
  });
}

/**
 * Styled alert dialog — replaces window.alert().
 */
function showAlert({ title, message }) {
  const footer = `
    <div style="display:flex;justify-content:flex-end;">
      <button class="btn btn-primary alert-ok-btn">${escapeHtml(t('common.ok') || 'OK')}</button>
    </div>
  `;
  const modal = showModal({
    title: title || t('common.info') || 'Info',
    content: `<div style="font-size:0.875rem;color:var(--text-secondary);margin:0;line-height:1.6;white-space:pre-wrap;word-break:break-word;">${escapeHtml(message)}</div>`,
    footer,
  });
  const okBtn = modal.overlay.querySelector('.alert-ok-btn');
  if (okBtn) okBtn.addEventListener('click', () => modal.close());
}
