import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# Configura las credenciales de la API de Spotify
client_credentials_manager = SpotifyClientCredentials(client_id='ee30ab7c7ed1447b96ae3313d0490f87', client_secret='b4d7773f3fdc439a99ba36e10f2b17eb')
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# Busca canciones de Taylor Swift
results = sp.search(q='artist:Taylor Swift', type='track')

# Reproduce la primera canción encontrada
if len(results['tracks']['items']) > 0:
    first_track_uri = results['tracks']['items'][0]['uri']
    sp.start_playback(uris=[first_track_uri])
