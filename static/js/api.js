/**
 * Central API paths — web UI and external clients use the same REST surface.
 * Base path is relative to the site origin (works behind reverse proxies if same host).
 */
window.IB = {
  index:    '/api',
  status:   '/api/status',
  chat:     '/api/chat',
  chatStream: '/api/chat/stream',
  analyze:  '/api/analyze',
  analyzeStream: '/api/analyze/stream',
};
