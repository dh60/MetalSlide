# MetalSlide
A slideshow app for macOS using Metal 4 and MetalFX for high quality image viewing.

![MetalSlide User Interface](demo.png)

## Features
- Displays photos from a folder in shuffled order
- Metal 4 pipeline with native rendering (no OS scaling)
- MetalFX upscaling, Jinc downscaling
- Auto-advance timer support

## Keybinds

| Key | Action |
|-----|--------|
| Right Arrow / Space | Next slide |
| Left Arrow | Previous slide |
| Delete | Trash current image |
| i | Toggle info overlay |
| 0-9 | Set auto-advance interval (seconds, 0 = off) |
| Esc | Quit |

## Requirements
- macOS 26+
- Apple Silicon
