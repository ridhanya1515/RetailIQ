// RetailIQ UI Interaction & Theme Management

document.addEventListener("DOMContentLoaded", () => {
    // 1. Theme Management
    const themeToggleBtn = document.getElementById("theme-toggle-btn");
    const currentTheme = localStorage.getItem("theme") || "dark";
    
    // Set initial theme
    document.documentElement.setAttribute("data-theme", currentTheme);
    updateThemeIcon(currentTheme);
    
    if (themeToggleBtn) {
        themeToggleBtn.addEventListener("click", () => {
            const theme = document.documentElement.getAttribute("data-theme");
            const newTheme = theme === "dark" ? "light" : "dark";
            
            document.documentElement.setAttribute("data-theme", newTheme);
            localStorage.setItem("theme", newTheme);
            updateThemeIcon(newTheme);
            
            // Trigger a redraw of Plotly charts to reflect background theme changes if necessary
            window.dispatchEvent(new Event('resize'));
        });
    }
    
    function updateThemeIcon(theme) {
        const icon = document.querySelector("#theme-toggle-btn i");
        if (icon) {
            if (theme === "light") {
                icon.className = "bi bi-moon-stars-fill";
            } else {
                icon.className = "bi bi-sun-fill";
            }
        }
    }

    // 2. Plotly Charts Responsiveness
    // Listen for window resize to force Plotly charts to adapt to width
    window.addEventListener('resize', () => {
        const plotlyCharts = document.querySelectorAll('.plotly-graph-div');
        plotlyCharts.forEach(chart => {
            if (typeof Plotly !== 'undefined') {
                Plotly.Plots.resize(chart);
            }
        });
    });

    // 3. Sidebar Responsive Toggle
    const sidebarCollapseBtn = document.getElementById("sidebarCollapse");
    const sidebar = document.getElementById("sidebar");
    if (sidebarCollapseBtn && sidebar) {
        sidebarCollapseBtn.addEventListener("click", () => {
            sidebar.classList.toggle("active");
        });
    }
});
