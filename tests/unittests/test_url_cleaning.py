#!/usr/bin/env python3

import pytest

from fraudcrawler.scraping.serp import SerpApi


@pytest.fixture
def ricardo_urls():
    """Test URLs from Ricardo with tracking parameters."""
    return [
        "https://www.ricardo.ch/it/a/party-cooler-50l-edelstahl-1258654784/?srsltid=AfmBOor1uTLRhTr9omRJOPPCGfzq0qSwlycUzQVu_w6LYzE3L8y_YL3I",
        "https://www.ricardo.ch/it/a/party-cooler-50l-edelstahl-1258654784/?srsltid=AfmBOorTWSb3cDNoyJjtrdXvma8Uie5RZ7yUf6X9lEL-O1-aFgt5EEjW",
        "https://www.ricardo.ch/it/a/party-cooler-50l-edelstahl-1258654784/?utm_source=google&utm_medium=cpc&srsltid=test",
    ]


@pytest.fixture
def ebay_urls():
    """Test URLs from eBay with tracking parameters."""
    return [
        "https://www.ebay.com/itm/123456?utm_source=test&other_param=value",
        "https://www.ebay.com/itm/789012?srsltid=tracking&utm_campaign=test",
        "https://www.ebay.com/itm/345678?param1=value1&param2=value2",
    ]


@pytest.fixture
def other_urls():
    """Test URLs from other domains with tracking parameters."""
    return [
        "https://www.amazon.com/product/123?utm_source=google&utm_medium=cpc",
        "https://www.galaxus.ch/de/product/456?srsltid=tracking&utm_term=test",
        "https://www.digitec.ch/fr/product/789?utm_campaign=test&other_param=value",
    ]


def test_remove_tracking_parameters_ricardo_urls(ricardo_urls):
    """Test that Ricardo URLs are cleaned correctly by removing tracking parameters."""
    expected_clean = "https://www.ricardo.ch/it/a/party-cooler-50l-edelstahl-1258654784/"
    
    for url in ricardo_urls:
        cleaned = SerpApi._remove_tracking_parameters(url)
        assert cleaned == expected_clean, f"Failed to clean URL: {url}"


def test_remove_tracking_parameters_ebay_urls(ebay_urls):
    """Test that eBay URLs have all query parameters removed."""
    expected_clean_urls = [
        "https://www.ebay.com/itm/123456",
        "https://www.ebay.com/itm/789012", 
        "https://www.ebay.com/itm/345678",
    ]
    
    for url, expected in zip(ebay_urls, expected_clean_urls):
        cleaned = SerpApi._remove_tracking_parameters(url)
        assert cleaned == expected, f"Failed to clean eBay URL: {url}"


def test_remove_tracking_parameters_other_urls(other_urls):
    """Test that other domain URLs have tracking parameters removed but keep other params."""
    expected_clean_urls = [
        "https://www.amazon.com/product/123",
        "https://www.galaxus.ch/de/product/456",
        "https://www.digitec.ch/fr/product/789?other_param=value",
    ]
    
    for url, expected in zip(other_urls, expected_clean_urls):
        cleaned = SerpApi._remove_tracking_parameters(url)
        assert cleaned == expected, f"Failed to clean other URL: {url}"


def test_remove_tracking_parameters_no_tracking():
    """Test URLs that don't have tracking parameters remain unchanged."""
    clean_urls = [
        "https://www.ricardo.ch/it/a/party-cooler-50l-edelstahl-1258654784/",
        "https://www.ebay.com/itm/123456",
        "https://www.amazon.com/product/123?param1=value1",
    ]
    
    for url in clean_urls:
        cleaned = SerpApi._remove_tracking_parameters(url)
        assert cleaned == url, f"Clean URL was modified: {url}"


def test_remove_tracking_parameters_edge_cases():
    """Test edge cases for URL cleaning."""
    test_cases = [
        # URL with only tracking parameters
        ("https://www.ricardo.ch/product/?srsltid=test", "https://www.ricardo.ch/product/"),
        # URL with mixed tracking and non-tracking parameters
        ("https://www.ricardo.ch/product/?param1=value1&srsltid=test&param2=value2", 
         "https://www.ricardo.ch/product/?param1=value1&param2=value2"),
        # URL with fragment
        ("https://www.ricardo.ch/product/?srsltid=test#section", "https://www.ricardo.ch/product/#section"),
        # URL with path parameters
        ("https://www.ricardo.ch/product/123/?srsltid=test", "https://www.ricardo.ch/product/123/"),
        # Empty URL
        ("", ""),
        # URL without scheme
        ("//www.ricardo.ch/product/?srsltid=test", "//www.ricardo.ch/product/"),
    ]
    
    for url, expected in test_cases:
        cleaned = SerpApi._remove_tracking_parameters(url)
        assert cleaned == expected, f"Failed to clean edge case URL: {url}"


def test_remove_tracking_parameters_known_trackers():
    """Test that all known tracking parameters are removed."""
    known_trackers = ["srsltid", "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content"]
    
    for tracker in known_trackers:
        url = f"https://www.ricardo.ch/product/?{tracker}=test_value"
        cleaned = SerpApi._remove_tracking_parameters(url)
        assert cleaned == "https://www.ricardo.ch/product/", f"Failed to remove tracker: {tracker}" 