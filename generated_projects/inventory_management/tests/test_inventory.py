import os
import pytest
from database import (
    init_db, add_product, update_product, delete_product,
    get_all_products, search_products, get_low_stock_products, get_inventory_stats
)

@pytest.fixture
def test_db(tmp_path):
    db_file = tmp_path / "test_inventory.db"
    db_path = str(db_file)
    init_db(db_path)
    return db_path

def test_add_and_get_product(test_db):
    add_product("P001", "Laptop", "Electronics", 10, 999.99, test_db)
    products = get_all_products(test_db)
    assert len(products) == 1
    assert products[0]["product_id"] == "P001"
    assert products[0]["name"] == "Laptop"
    assert products[0]["quantity"] == 10
    assert products[0]["price"] == 999.99

def test_duplicate_product_id(test_db):
    add_product("P001", "Laptop", "Electronics", 10, 999.99, test_db)
    with pytest.raises(ValueError):
        add_product("P001", "Another Laptop", "Electronics", 5, 899.99, test_db)

def test_negative_quantity_and_price(test_db):
    with pytest.raises(ValueError):
        add_product("P002", "Mouse", "Electronics", -1, 19.99, test_db)
    with pytest.raises(ValueError):
        add_product("P003", "Keyboard", "Electronics", 5, -10.00, test_db)

def test_update_product(test_db):
    add_product("P001", "Laptop", "Electronics", 10, 999.99, test_db)
    update_product("P001", "Gaming Laptop", "Electronics", 8, 1299.99, test_db)
    products = get_all_products(test_db)
    assert products[0]["name"] == "Gaming Laptop"
    assert products[0]["quantity"] == 8
    assert products[0]["price"] == 1299.99

def test_delete_product(test_db):
    add_product("P001", "Laptop", "Electronics", 10, 999.99, test_db)
    delete_product("P001", test_db)
    products = get_all_products(test_db)
    assert len(products) == 0

def test_search_products(test_db):
    add_product("P001", "MacBook Pro", "Electronics", 5, 1999.99, test_db)
    add_product("P002", "Dell XPS", "Electronics", 3, 1499.99, test_db)
    add_product("P003", "Desk Chair", "Furniture", 10, 150.00, test_db)

    results = search_products("Book", test_db)
    assert len(results) == 1
    assert results[0]["product_id"] == "P001"

    results_cat = search_products("Furniture", test_db)
    assert len(results_cat) == 1
    assert results_cat[0]["product_id"] == "P003"

def test_low_stock_and_stats(test_db):
    add_product("P001", "Item A", "Cat1", 2, 50.00, test_db)
    add_product("P002", "Item B", "Cat1", 10, 20.00, test_db)

    low_stock = get_low_stock_products(threshold=5, db_path=test_db)
    assert len(low_stock) == 1
    assert low_stock[0]["product_id"] == "P001"

    stats = get_inventory_stats(test_db)
    assert stats["total_products"] == 2
    assert stats["total_quantity"] == 12
    assert stats["total_value"] == (2 * 50.00) + (10 * 20.00)
