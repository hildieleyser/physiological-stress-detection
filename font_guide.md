# How to Install and Use Different Fonts in Python Matplotlib

## 🔤 **Font Installation Methods:**

### **Method 1: System Font Installation (Recommended)**
1. **Download the font** (e.g., Lexend from Google Fonts)
2. **Install on your system:**
   - **macOS:** Double-click font file → Install Font
   - **Windows:** Right-click font file → Install
   - **Linux:** Copy to `~/.fonts/` or `/usr/share/fonts/`

### **Method 2: Direct Font File Usage**
```python
import matplotlib.font_manager as fm

# Add font file directly
font_path = '/path/to/your/font.ttf'
custom_font = fm.FontProperties(fname=font_path)

# Use in plots
plt.title('Your Title', fontproperties=custom_font)
```

### **Method 3: Font Installation via Python**
```python
import matplotlib.font_manager as fm
import os

# Download and install font
font_url = 'https://github.com/google/fonts/raw/main/ofl/lexend/Lexend-Regular.ttf'
font_path = fm.findfont(fm.FontProperties(family='Lexend'))

# If font not found, download it
if 'DejaVuSans' in font_path:  # Default fallback
    # Download and install font
    import urllib.request
    urllib.request.urlretrieve(font_url, 'Lexend-Regular.ttf')
    fm.fontManager.addfont('Lexend-Regular.ttf')
```

## 🎨 **Font Usage in Matplotlib:**

### **Check Available Fonts:**
```python
import matplotlib.font_manager as fm

# List all available fonts
fonts = [f.name for f in fm.fontManager.ttflist]
print('Available fonts:', sorted(set(fonts)))

# Check specific font
print('Arial available:', 'Arial' in fonts)
print('Helvetica available:', 'Helvetica' in fonts)
print('Lexend available:', 'Lexend' in fonts)
```

### **Set Font Globally:**
```python
import matplotlib.pyplot as plt

# Method 1: Using font family
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 14
})

# Method 2: Using specific font
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 14
})
```

### **Set Font for Specific Elements:**
```python
# For titles
plt.title('Your Title', fontfamily='Arial', fontsize=16, fontweight='bold')

# For axis labels
plt.xlabel('X Label', fontfamily='Arial', fontsize=14)

# For legend
plt.legend(fontsize=12, prop={'family': 'Arial'})
```

## 📝 **Popular Fonts for Presentations:**

### **Professional Fonts:**
- **Arial** - Clean, professional, widely available
- **Helvetica** - Modern, clean, excellent readability
- **Calibri** - Microsoft's default, very readable
- **Roboto** - Google's font, modern and clean

### **Academic Fonts:**
- **Times New Roman** - Traditional academic
- **Garamond** - Elegant, classic
- **Palatino** - Readable serif font

### **Modern Fonts:**
- **Lexend** - Designed for readability
- **Inter** - Modern, clean
- **Source Sans Pro** - Adobe's clean font

## 🔧 **Font Troubleshooting:**

### **Font Not Found:**
```python
# Check what font is actually being used
import matplotlib.font_manager as fm
print('Current font:', fm.findfont(fm.FontProperties(family='Arial')))
```

### **Font Fallbacks:**
```python
# Always provide fallbacks
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans', 'Liberation Sans'],
})
```

### **Force Font Refresh:**
```python
# After installing new fonts
import matplotlib.font_manager as fm
fm._rebuild()
plt.rcParams.update(plt.rcParamsDefault)
```

## 🎯 **Best Practices:**

1. **Always provide fallbacks** - not all systems have the same fonts
2. **Test on target system** - fonts may render differently
3. **Use consistent fonts** throughout your presentation
4. **Consider readability** - some fonts look good but are hard to read
5. **Check font licensing** - some fonts require licenses for commercial use

## 📊 **Example with Multiple Fonts:**
```python
import matplotlib.pyplot as plt
import numpy as np

# Set up fonts
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 14
})

# Create plot
x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y, linewidth=2, color='#190B28')
plt.title('Sample Plot with Custom Font', fontsize=20, fontweight='bold')
plt.xlabel('X Axis', fontsize=16)
plt.ylabel('Y Axis', fontsize=16)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

This guide should help you install and use any font you want in your Python matplotlib plots! 