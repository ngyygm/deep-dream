/* ==========================================
   i18n Engine
   ========================================== */
window.I18N = {
  currentLang: localStorage.getItem('deepdream_lang') || localStorage.getItem('tmg_lang') || 'zh',
  fallbackLang: 'zh',
  langData: {},

  init() {
    // Register all loaded language packs
    if (window.I18N_LANG_ZH) this.langData['zh'] = window.I18N_LANG_ZH;
    if (window.I18N_LANG_EN) this.langData['en'] = window.I18N_LANG_EN;
    if (window.I18N_LANG_JA) this.langData['ja'] = window.I18N_LANG_JA;

    // Apply saved language
    this.applyLang(this.currentLang);

    // Wire language selector
    const sel = document.getElementById('lang-selector');
    if (sel) {
      sel.value = this.currentLang;
      sel.addEventListener('change', () => this.setLang(sel.value));
    }
  },

  setLang(lang) {
    if (!this.langData[lang]) return;
    this.currentLang = lang;
    localStorage.setItem('deepdream_lang', lang);
    this.applyLang(lang);

    // Re-render current page
    if (typeof handleRoute === 'function') handleRoute();
  },

  applyLang(lang) {
    // Update all data-i18n elements
    document.querySelectorAll('[data-i18n]').forEach(el => {
      const key = el.getAttribute('data-i18n');
      const text = this.t(key);
      if (text) el.textContent = text;
    });

    // Update page title
    const pageTitle = document.getElementById('page-title');
    if (pageTitle && typeof _pageTitleKeys !== 'undefined') {
      const page = (window.location.hash || '#dashboard').slice(1).split('/')[0] || 'dashboard';
      const key = _pageTitleKeys[page] || '';
      pageTitle.textContent = this.t(key) || page;
    }
  },

  t(key, params) {
    let text = (this.langData[this.currentLang] && this.langData[this.currentLang][key])
      || (this.langData[this.fallbackLang] && this.langData[this.fallbackLang][key])
      || key;

    // Interpolate params like {count}, {name}, etc.
    if (params) {
      Object.keys(params).forEach(k => {
        text = text.replace(new RegExp('\\{' + k + '\\}', 'g'), params[k]);
      });
    }
    return text;
  },
};

// Shorthand
window.t = (key, params) => window.I18N.t(key, params);
