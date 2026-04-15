// Navigation Logic
document.addEventListener('DOMContentLoaded', () => {
    
    // Tab switching
    const navItems = document.querySelectorAll('.nav-item');
    const tabPanes = document.querySelectorAll('.tab-pane');
    const pageTitle = document.getElementById('page-title');

    navItems.forEach(item => {
        item.addEventListener('click', () => {
            // Update active nav
            navItems.forEach(n => n.classList.remove('active'));
            item.classList.add('active');

            // Update title
            pageTitle.textContent = item.textContent.trim();

            // Show tab
            const targetId = item.getAttribute('data-tab');
            tabPanes.forEach(pane => {
                pane.classList.remove('active');
                if (pane.id === targetId) pane.classList.add('active');
            });

            // Lazy load data based on tab
            if (targetId === 'dashboard-tab' && !window.dashboardLoaded) loadDashboard();
            if (targetId === 'analyze-tab' && !window.analyzeLoaded) loadAnalyze();
            if (targetId === 'fake-users-tab' && !window.fakeUsersLoaded) loadFakeUsers(1);
        });
    });

    // Initial Load
    loadDashboard();
});

// APIs
async function loadDashboard() {
    try {
        const res = await fetch('/dashboard');
        const data = await res.json();
        
        if (data.status === 'ok') {
            document.getElementById('dashboard-loader').classList.add('hidden');
            document.getElementById('dashboard-content').classList.remove('hidden');
            window.dashboardLoaded = true;

            const d = data.detection;
            const s = data.graph_stats;

            // KPIs
            document.getElementById('kpi-total-nodes').textContent = s.num_nodes.toLocaleString();
            document.getElementById('kpi-total-edges').textContent = s.num_edges.toLocaleString() + ' edges loaded';
            
            document.getElementById('kpi-real-nodes').textContent = d.real_count.toLocaleString();
            document.getElementById('kpi-fake-nodes').textContent = d.fake_count.toLocaleString();
            document.getElementById('kpi-fake-percentage').textContent = d.fake_percentage + '% of network';

            // Averages
            const formatDec = val => val ? val.toFixed(4) : "0.0000";
            document.getElementById('pr-real').textContent = formatDec(d.avg_pagerank_real);
            document.getElementById('pr-fake').textContent = formatDec(d.avg_pagerank_fake);
            document.getElementById('clus-real').textContent = formatDec(d.avg_clustering_real);
            document.getElementById('clus-fake').textContent = formatDec(d.avg_clustering_fake);
        }
    } catch (e) {
        console.error("Dashboard failed:", e);
        document.getElementById('dashboard-loader').textContent = "Error loading dashboard data.";
    }
}

async function loadAnalyze() {
    try {
        const res = await fetch('/analyze');
        const data = await res.json();
        
        if (data.status === 'ok') {
            document.getElementById('analyze-loader').classList.add('hidden');
            document.getElementById('analyze-content').classList.remove('hidden');
            window.analyzeLoaded = true;

            const m = data.ml_metrics;
            const container = document.getElementById('ml-metrics-container');
            
            if (m.note) {
                container.innerHTML = `<div class="metric-pill"><span>Status</span><strong>${m.note}</strong></div>`;
            } else {
                container.innerHTML = `
                    <div class="metric-pill"><span>Accuracy</span><strong>${(m.accuracy * 100).toFixed(1)}%</strong></div>
                    <div class="metric-pill"><span>Precision</span><strong>${(m.precision * 100).toFixed(1)}%</strong></div>
                    <div class="metric-pill"><span>Recall</span><strong>${(m.recall * 100).toFixed(1)}%</strong></div>
                    <div class="metric-pill"><span>F1-Score</span><strong>${(m.f1_score * 100).toFixed(1)}%</strong></div>
                `;
            }
        }
    } catch (e) {
        console.error("Analyze failed:", e);
        document.getElementById('analyze-loader').textContent = "Error loading ML metrics.";
    }
}

let currentPage = 1;
const pageSize = 50;

async function loadFakeUsers(page) {
    const tbody = document.getElementById('fake-users-tbody');
    tbody.innerHTML = '<tr><td colspan="5" class="text-center loader-row">Loading users...</td></tr>';
    
    try {
        const res = await fetch(`/fake-users?page=${page}&page_size=${pageSize}`);
        const data = await res.json();
        
        if (data.status === 'ok') {
            window.fakeUsersLoaded = true;
            currentPage = data.page;
            
            document.getElementById('page-indicator').textContent = `Page ${data.page} / ${data.total_pages}`;
            
            document.getElementById('btn-prev').disabled = data.page <= 1;
            document.getElementById('btn-next').disabled = data.page >= data.total_pages;

            tbody.innerHTML = '';
            
            if (data.fake_accounts.length === 0) {
                tbody.innerHTML = '<tr><td colspan="5" class="text-center">No fake accounts detected!</td></tr>';
                return;
            }

            data.fake_accounts.forEach(user => {
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td><strong>#${user.node}</strong></td>
                    <td class="text-red">${user.out_degree}</td>
                    <td>${user.in_degree}</td>
                    <td><span class="tag">${(user.ml_probability * 100).toFixed(1)}%</span></td>
                    <td class="text-muted"><small>${user.reason}</small></td>
                `;
                tbody.appendChild(tr);
            });
        }
    } catch (e) {
        console.error("Fake users failed:", e);
        tbody.innerHTML = '<tr><td colspan="5" class="text-center text-red">Error loading users.</td></tr>';
    }
}

// Pagination bindings
document.getElementById('btn-prev').addEventListener('click', () => {
    if (currentPage > 1) loadFakeUsers(currentPage - 1);
});
document.getElementById('btn-next').addEventListener('click', () => {
    loadFakeUsers(currentPage + 1);
});
