
#!/usr/bin/with-contenv bashio

# Get the key from HA config
OPENAI_API_KEY=$(bashio::config 'openai_api_key')

# Inject it into config.ini
sed -i "s|api_key = .*|api_key = ${OPENAI_API_KEY}|" /app/config.ini

# Start TARS
python3 app.py