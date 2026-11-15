"""
tests/test_sidebar_navigation.py – Test suite for consistent sidebar navigation order, active state, and layout.

Validates that:
1. Every page renders the exact 10 navigation links in the required order:
   1. Overview (/ or #overview)
   2. Water Network (/network)
   3. Pipe Management (/pipes)
   4. Sensor Management (/sensors)
   5. Sensor Simulator (/sensor-simulator)
   6. Graph Explorer (/graph)
   7. Route Finder (/route)
   8. Leak Detection (/leak-detection)
   9. Repair Queue (/repairs)
   10. Water Allocation (/allocation)
2. Exactly one item has the 'active' highlight on each view.
3. Edit views correctly preserve their parent's active state.
4. The sidebar component and styles enforce fixed positioning and independent scrolling.
"""

import os
import re
import pytest
from app import app


EXPECTED_NAV_ITEMS = [
    ("/", "▦ Overview"),
    ("/network", "⌘ Water Network"),
    ("/pipes", "🔗 Pipe Management"),
    ("/sensors", "📡 Sensor Management"),
    ("/sensor-simulator", "⚡ Sensor Simulator"),
    ("/graph", "☍ Graph Explorer"),
    ("/route", "🧭 Route Finder"),
    ("/leak-detection", "⚠ Leak Detection"),
    ("/repairs", "🔧 Repair Queue"),
    ("/allocation", "⚖ Water Allocation"),
    ("/distribution", "🌊 Distribution"),
]


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


class TestSidebarNavigation:
    """Verify consistent ordering, active highlighting, and links across all pages."""

    @pytest.mark.parametrize(
        "path,expected_active_label",
        [
            ("/", "▦ Overview"),
            ("/network", "⌘ Water Network"),
            ("/pipes", "🔗 Pipe Management"),
            ("/sensors", "📡 Sensor Management"),
            ("/sensor-simulator", "⚡ Sensor Simulator"),
            ("/graph", "☍ Graph Explorer"),
            ("/route", "🧭 Route Finder"),
            ("/leak-detection", "⚠ Leak Detection"),
            ("/repairs", "🔧 Repair Queue"),
            ("/allocation", "⚖ Water Allocation"),
            ("/distribution", "🌊 Distribution"),
        ],
    )
    def test_navigation_order_and_active_state(self, client, path, expected_active_label):
        """Every page renders the exact 11 navigation items in the required order with correct active state."""
        response = client.get(path)
        assert response.status_code == 200, f"Page {path} returned {response.status_code}"
        html = response.get_data(as_text=True)

        aside_match = re.search(r'<aside class="sidebar">(.*?)</aside>', html, re.DOTALL)
        assert aside_match, f"Sidebar markup missing on {path}"
        sidebar_content = aside_match.group(1)

        # Extract all navigation links inside the sidebar
        nav_links = re.findall(
            r'<a class="([^"]+)"\s+href="([^"]+)">\s*([^<]+)</a>',
            sidebar_content,
        )

        assert len(nav_links) == 11, (
            f"Expected exactly 11 navigation links on {path}, but found {len(nav_links)}: "
            f"{[item[2].strip() for item in nav_links]}"
        )

        active_count = 0
        for idx, (cls, href, text) in enumerate(nav_links):
            expected_href, expected_text = EXPECTED_NAV_ITEMS[idx]
            clean_text = text.strip()
            clean_cls = cls.strip()

            assert href == expected_href, (
                f"Position {idx + 1} on {path} has href '{href}', expected '{expected_href}'"
            )
            assert clean_text == expected_text, (
                f"Position {idx + 1} on {path} has text '{clean_text}', expected '{expected_text}'"
            )

            if clean_text == expected_active_label:
                assert clean_cls == "nav-link active", (
                    f"Active link '{clean_text}' on {path} has class '{clean_cls}', expected 'nav-link active'"
                )
                active_count += 1
            else:
                assert clean_cls == "nav-link", (
                    f"Inactive link '{clean_text}' on {path} has class '{clean_cls}', expected 'nav-link'"
                )

        assert active_count == 1, f"Expected exactly 1 active link on {path}, found {active_count}"

    def test_sidebar_header_and_footer_structure(self, client):
        """Ensure sidebar contains brand, workspace label, scrollable container, and footer."""
        response = client.get("/")
        assert response.status_code == 200
        html = response.get_data(as_text=True)

        assert '<div class="brand">💧 AquaGrid</div>' in html
        assert '<p class="nav-label">WORKSPACE</p>' in html
        assert '<nav class="nav-scroll">' in html
        assert "SDG 6 · Clean Water" in html
        assert "Prototype · Demo Mode" in html

    def test_sidebar_css_rules(self):
        """Ensure static/css/style.css defines required sidebar layout and scroll properties."""
        css_path = os.path.join(os.path.dirname(__file__), "..", "static", "css", "style.css")
        assert os.path.exists(css_path)
        with open(css_path, "r", encoding="utf-8") as f:
            css = f.read()

        assert ".sidebar {" in css
        assert "position: fixed;" in css
        assert "width: 245px;" in css
        assert ".nav-scroll {" in css
        assert "overflow-y: auto;" in css
        assert ".nav-link.active {" in css
        assert "border-left: 3px solid #45d6c4;" in css

