create database exercise2;
\c exercise2

CREATE TABLE customers (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    email VARCHAR(255) NOT NULL,
    phone VARCHAR(50),
    address VARCHAR(255)
);

CREATE TABLE categories (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL
);

CREATE TABLE products (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    category_id INT,
    price DECIMAL(10, 2) NOT NULL,
    FOREIGN KEY (category_id) REFERENCES categories(id)
);

CREATE TABLE orders (
    id SERIAL PRIMARY KEY,
    product_id INT,
    customer_id INT,
    quantity INT NOT NULL,
    price DECIMAL(10, 2) NOT NULL,
    order_date TIMESTAMP NOT NULL,
    FOREIGN KEY (product_id) REFERENCES products(id),
    FOREIGN KEY (customer_id) REFERENCES customers(id)
);

CREATE TABLE product_reviews (
    id SERIAL PRIMARY KEY,
    product_id INT,
    customer_id INT,
    rating INT NOT NULL,
    review_date TIMESTAMP NOT NULL,
    FOREIGN KEY (product_id) REFERENCES products(id),
    FOREIGN KEY (customer_id) REFERENCES customers(id)
);

-- Insert sample data into customers
INSERT INTO customers (name, email, phone, address) VALUES
('John Doe', 'john.doe@example.com', '123-456-7890', '123 Elm Street'),
('Jane Smith', 'jane.smith@example.com', '234-567-8901', '456 Oak Avenue'),
('Alice Johnson', 'alice.johnson@example.com', '345-678-9012', '789 Pine Road'),
('Bob Brown', 'bob.brown@example.com', '456-789-0123', '101 Maple Lane'),
('Charlie Davis', 'charlie.davis@example.com', '567-890-1234', '202 Birch Boulevard');

-- Insert sample data into categories
INSERT INTO categories (name) VALUES
('Electronics'),
('Books'),
('Clothing');

-- Insert sample data into products
INSERT INTO products (name, category_id, price) VALUES
('Smartphone', 1, 699.99),
('Laptop', 1, 999.99),
('Novel', 2, 19.99),
('T-Shirt', 3, 9.99),
('Jeans', 3, 49.99);

-- Insert sample data into orders
INSERT INTO orders (product_id, customer_id, quantity, price, order_date) VALUES
(1, 1, 1, 699.99, '2023-01-15 10:00:00'),
(2, 1, 1, 999.99, '2023-01-20 15:30:00'),
(3, 2, 2, 39.98, '2023-02-10 09:45:00'),
(4, 3, 3, 29.97, '2023-02-15 14:15:00'),
(5, 4, 1, 49.99, '2023-03-05 11:00:00');

-- Insert sample data into product_reviews
INSERT INTO product_reviews (product_id, customer_id, rating, review_date) VALUES
(1, 1, 5, '2023-01-16 08:00:00'),
(2, 1, 4, '2023-01-21 13:00:00'),
(3, 2, 3, '2023-02-11 10:00:00'),
(4, 3, 4, '2023-02-16 15:00:00'),
(5, 4, 2, '2023-03-06 09:00:00');

--List all products in a specific category. 
SELECT p.id, p.name, p.price
FROM products p
JOIN categories c ON p.category_id = c.id
WHERE c.name = 'Books';

--Find the top 5 customers who have placed the most orders.
SELECT c.id, c.name, COUNT(o.id) AS order_count
FROM customers c
JOIN orders o ON c.id = o.customer_id
GROUP BY c.id, c.name
ORDER BY order_count DESC
LIMIT 5;

--retrieve the order details for a specific customer, including product names and quantities.
SELECT o.id AS order_id, p.name AS product_name, o.quantity, o.price, o.order_date
FROM orders o
JOIN products p ON o.product_id = p.id
WHERE o.customer_id = 1;

--Get the average rating for each product.
SELECT p.id, p.name, AVG(r.rating) AS average_rating
FROM products p
JOIN product_reviews r ON p.id = r.product_id
GROUP BY p.id, p.name;

--Find all customers who have not placed any orders in the last 6/12 months.
SELECT c.id, c.name
FROM customers c
LEFT JOIN orders o ON c.id = o.customer_id AND o.order_date >= NOW() - INTERVAL '6 months'
WHERE o.id IS NULL;


select p.name,c.name from categories c join products p on c.id = p.category_id;