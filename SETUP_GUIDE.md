# JazzVibe-AI Flask Setup Guide

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Flask Application

```bash
python app_new.py
```

The application will start at `http://localhost:5000`

## 📁 Project Structure

```
JazzVibe-AI/
├── app_new.py              # Flask application server
├── templates/
│   └── index.html          # Main HTML template
├── static/
│   ├── css/
│   │   └── style.css       # Premium UI styling
│   ├── js/
│   │   └── main.js         # Interactive JavaScript
│   └── outputs/           # Generated MIDI files
└── requirements.txt        # Python dependencies
```

## 🎨 Features

- **Premium UI**: Dark glassmorphism design with smooth animations
- **Fast Generation**: CPU-friendly music generation using jazz patterns
- **Customizable**: Adjust note count and creativity level
- **Instant Download**: Download generated MIDI files immediately
- **Responsive**: Works on all screen sizes

## ⚙️ Configuration

### Port Configuration

Edit `app_new.py` to change the default port:

```python
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)  # Change port here
```

### Generation Parameters

- **Number of Notes**: 100-1000 (default: 500)
- **Creativity Level**: 0.5-2.0 (default: 1.0)
  - Lower values = more conservative
  - Higher values = more creative

## 🔧 API Endpoints

### `GET /`
- Renders the main page

### `POST /generate`
- Generates jazz music
- **Request Body**:
  ```json
  {
    "num_notes": 500,
    "temperature": 1.0
  }
  ```
- **Response**:
  ```json
  {
    "success": true,
    "filename": "jazz_20240109_123456_abc123.mid",
    "message": "Jazz music generated successfully!"
  }
  ```

### `GET /outputs/<filename>`
- Serves generated MIDI files for playback

### `GET /download/<filename>`
- Downloads generated MIDI files

## 🎵 Music Generation

The current implementation uses a probabilistic jazz pattern generator that:

1. Uses jazz-friendly scales (C, F, Bb)
2. Incorporates common jazz chord progressions
3. Balances single notes and chords (70/30 ratio)
4. Generates realistic jazz melodies

## 🌐 Deployment

### Production Deployment

For production, use a WSGI server like Gunicorn:

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app_new:app
```

### Environment Variables

Set the following environment variables for production:

```bash
export FLASK_ENV=production
export FLASK_DEBUG=0
```

## 🛠️ Troubleshooting

### Port Already in Use

If port 5000 is already in use, change it in `app_new.py`:

```python
app.run(debug=True, host='0.0.0.0', port=5001)
```

### MIDI Files Not Playing

Some browsers may not support MIDI playback natively. Use a MIDI player application or convert to MP3 using online tools.

### Music21 Installation Issues

If you encounter issues with music21:

```bash
pip install --upgrade music21
```

## 📝 Customization

### Adding New Scales

Edit `SimpleJazzGenerator` in `app_new.py`:

```python
self.jazz_scales = {
    'C': ['C', 'D', 'E', 'F', 'G', 'A', 'B'],
    'D': ['D', 'E', 'F#', 'G', 'A', 'B', 'C#'],  # Add new scale
}
```

### Changing Chord Progressions

Edit the `common_chords` list:

```python
self.common_chords = [
    ['C', 'E', 'G'],      # C major
    ['F', 'A', 'C'],      # F major
    # Add your custom chords
]
```

## 🔒 Security

For production deployment:

1. Disable debug mode
2. Use environment variables for sensitive data
3. Implement rate limiting
4. Add authentication if needed
5. Use HTTPS

## 📊 Performance

The current implementation is optimized for CPU usage:
- Fast generation (< 5 seconds)
- Low memory footprint
- No GPU required
- Suitable for shared hosting

## 🤝 Contributing

To enhance the music generation:

1. Replace `SimpleJazzGenerator` with a pre-trained model
2. Add more sophisticated jazz patterns
3. Implement user authentication
4. Add database for saving user compositions
5. Create mobile app version

## 📞 Support

For issues and questions:
- Check the troubleshooting section
- Review the code documentation
- Test with different parameters

---

**Built with ❤️ using Flask & Music21**

*© 2024 JazzVibe-AI | Premium AI Music Generation*
