/** @type {import('tailwindcss').Config} */
module.exports = {
  darkMode: 'class',
  content: [
    './core/server/static/index.html',
    './core/server/static/js/**/*.js',
    './core/server/static/css/**/*.css',
  ],
  theme: {
    extend: {
      colors: {
        surface: '#1c1520',
        'surface-hover': '#2a1e28',
        'bg-input': '#170f14',
      }
    }
  },
  plugins: [],
}
