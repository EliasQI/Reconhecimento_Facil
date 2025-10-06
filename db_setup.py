import sqlite3

conn = sqlite3.connect('faces.db')
cursor = conn.cursor()

cursor.execute('''
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    permissions TEXT,
    created_at TEXT
)
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS embeddings (
    user_id TEXT,
    vector BLOB,
    FOREIGN KEY(user_id) REFERENCES users(id)
)
''')

conn.commit()
conn.close()
print("✅ Banco criado com sucesso: faces.db")