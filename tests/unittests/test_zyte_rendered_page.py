import httpx
import pytest

from fraudcrawler.scraping.zyte import ZyteAPI


@pytest.mark.asyncio
async def test_fetch_rendered_page_normalizes_product_list():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "browserHtml": "<html><body>ok</body></html>",
                "statusCode": 207,
                "actions": [
                    {"action": "click", "status": "success"},
                    {"action": "waitFor", "status": "failed", "error": "timeout"},
                ],
                "networkCapture": [
                    {
                        "url": "https://shop.test/api/products",
                        "statusCode": 200,
                        "contentType": "application/json",
                        "responseBody": '{"items":[1]}',
                    }
                ],
                "productList": {
                    "products": [
                        {
                            "url": "/p/123",
                            "title": "Product Title",
                            "price": "CHF 99",
                            "images": ["https://img.test/a.jpg"],
                        }
                    ]
                },
            },
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        zyte = ZyteAPI(http_client=client, api_key="test-key", redis_use_cache=False)
        result = await zyte.fetch_rendered_page(url="https://shop.test/list")

    assert result.status_code == 207
    assert len(result.product_list_items) == 1
    assert result.product_list_items[0].name == "Product Title"
    assert result.action_statuses == ["click:success", "waitFor:failed"]
    assert result.action_error == "timeout"


@pytest.mark.asyncio
async def test_fetch_rendered_page_raises_for_empty_browser_html():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"browserHtml": "   "})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        zyte = ZyteAPI(http_client=client, api_key="test-key", redis_use_cache=False)
        with pytest.raises(httpx.HTTPError, match="empty browserHtml"):
            await zyte.fetch_rendered_page(url="https://shop.test/list")
