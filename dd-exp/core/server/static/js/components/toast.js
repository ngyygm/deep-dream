/* ==========================================
   Toast Notification Component
   ========================================== */

function createToastContainer() {
  let container = document.getElementById('toast-container');
  if (!container) {
    container = document.createElement('div');
    container.id = 'toast-container';
    container.className = 'toast-container';
    document.body.appendChild(container);
  }
  return container;
}

function showToast(message, type = 'info', duration = 4000) {
  const container = createToastContainer();
  const toast = document.createElement('div');
  toast.className = `toast toast-${type}`;

  const iconMap = {
    success: 'check-circle',
    error: 'x-circle',
    warning: 'alert-triangle',
    info: 'info',
  };

  toast.innerHTML = `
    <i data-lucide="${iconMap[type] || 'info'}" style="width:18px;height:18px;flex-shrink:0;margin-top:1px;color:var(--${type === 'error' ? 'error' : type === 'warning' ? 'warning' : type === 'success' ? 'success' : 'info'})"></i>
    <div style="flex:1;min-width:0;">
      <div style="word-break:break-word;">${escapeHtml(message)}</div>
    </div>
    <button style="background:none;border:none;color:var(--text-muted);cursor:pointer;padding:0;flex-shrink:0;" onclick="this.closest('.toast').remove()">
      <i data-lucide="x" style="width:14px;height:14px;"></i>
    </button>
  `;

  container.appendChild(toast);
  if (window.lucide) lucide.createIcons({ nodes: [toast] });

  if (duration > 0) {
    setTimeout(() => {
      toast.classList.add('removing');
      setTimeout(() => toast.remove(), 150);
    }, duration);
  }

  return toast;
}
