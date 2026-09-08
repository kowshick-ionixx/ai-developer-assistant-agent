import streamlit as st
import pandas as pd
from database import (
    init_db, add_product, update_product, delete_product,
    get_all_products, search_products, get_low_stock_products, get_inventory_stats
)

st.set_page_config(page_title="Inventory Management System", page_icon="📦", layout="wide")

# Initialize database
init_db()

st.title("📦 Inventory Management System")

# Sidebar navigation
st.sidebar.title("Navigation")
menu = st.sidebar.radio("Go to", ["Dashboard", "Manage Products", "Search & Filter", "Low Stock Alerts"])

# Dashboard
if menu == "Dashboard":
    st.header("📊 Dashboard Overview")
    stats = get_inventory_stats()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Total Unique Products", value=stats["total_products"])
    with col2:
        st.metric(label="Total Inventory Quantity", value=stats["total_quantity"])
    with col3:
        st.metric(label="Total Inventory Value ($)", value=f"${stats['total_value']:,.2f}")
    
    st.divider()
    st.subheader("Low Stock Warning (Quantity <= 5)")
    low_stock = get_low_stock_products(threshold=5)
    if low_stock:
        df_low = pd.DataFrame(low_stock)
        st.dataframe(df_low, use_container_width=True)
    else:
        st.success("All products are well-stocked!")

# Manage Products
elif menu == "Manage Products":
    st.header("🛠️ Product Management")
    
    tab1, tab2, tab3 = st.tabs(["Add Product", "Edit Product", "Delete Product"])
    
    with tab1:
        st.subheader("Add New Product")
        with st.form("add_product_form"):
            prod_id = st.text_input("Product ID")
            name = st.text_input("Product Name")
            category = st.text_input("Category")
            quantity = st.number_input("Quantity", min_value=0, step=1, value=0)
            price = st.number_input("Price ($)", min_value=0.0, format="%.2f", value=0.0)
            submitted = st.form_submit_button("Add Product")
            
            if submitted:
                if not prod_id or not name or not category:
                    st.error("Please fill in all required fields.")
                else:
                    try:
                        add_product(prod_id, name, category, int(quantity), float(price))
                        st.success(f"Product '{name}' added successfully!")
                    except Exception as e:
                        st.error(str(e))
                        
    with tab2:
        st.subheader("Edit Existing Product")
        products = get_all_products()
        if not products:
            st.info("No products available to edit.")
        else:
            prod_ids = [p["product_id"] for p in products]
            selected_id = st.selectbox("Select Product ID to Edit", prod_ids)
            
            prod_data = next((p for p in products if p["product_id"] == selected_id), None)
            
            if prod_data:
                with st.form("edit_product_form"):
                    name = st.text_input("Product Name", value=prod_data["name"])
                    category = st.text_input("Category", value=prod_data["category"])
                    quantity = st.number_input("Quantity", min_value=0, step=1, value=prod_data["quantity"])
                    price = st.number_input("Price ($)", min_value=0.0, format="%.2f", value=float(prod_data["price"]))
                    update_submitted = st.form_submit_button("Update Product")
                    
                    if update_submitted:
                        try:
                            update_product(selected_id, name, category, int(quantity), float(price))
                            st.success(f"Product '{selected_id}' updated successfully!")
                        except Exception as e:
                            st.error(str(e))
                            
    with tab3:
        st.subheader("Delete Product")
        products = get_all_products()
        if not products:
            st.info("No products available to delete.")
        else:
            prod_ids = [p["product_id"] for p in products]
            selected_id = st.selectbox("Select Product ID to Delete", prod_ids, key="delete_select")
            
            if st.button("Delete Selected Product", type="primary"):
                try:
                    delete_product(selected_id)
                    st.success(f"Product '{selected_id}' deleted successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(str(e))
                    
    st.divider()
    st.subheader("Current Inventory List")
    all_prods = get_all_products()
    if all_prods:
        st.dataframe(pd.DataFrame(all_prods), use_container_width=True)
    else:
        st.info("Inventory is currently empty.")

# Search & Filter
elif menu == "Search & Filter":
    st.header("🔍 Search Products")
    query = st.text_input("Search by Product Name or Category")
    if query:
        results = search_products(query)
        st.subheader(f"Found {len(results)} result(s)")
        if results:
            st.dataframe(pd.DataFrame(results), use_container_width=True)
        else:
            st.info("No matching products found.")
    else:
        st.info("Enter a search term to find products.")

# Low Stock Alerts
elif menu == "Low Stock Alerts":
    st.header("⚠️ Low Stock Inventory")
    threshold = st.slider("Low Stock Threshold", min_value=1, max_value=20, value=5)
    low_stock = get_low_stock_products(threshold=threshold)
    if low_stock:
        st.warning(f"Found {len(low_stock)} product(s) at or below quantity {threshold}.")
        st.dataframe(pd.DataFrame(low_stock), use_container_width=True)
    else:
        st.success(f"No products found with quantity <= {threshold}.")
