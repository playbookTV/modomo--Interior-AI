/**
 * BNA UI Global Theme Configuration
 * ReRoom design system globals
 */

export const globals = {
  // Typography scale
  typography: {
    fontFamily: {
      regular: 'System',
      medium: 'System',
      semiBold: 'System',
      bold: 'System',
    },
    fontSize: {
      xs: 12,
      sm: 14,
      base: 16,
      lg: 18,
      xl: 20,
      '2xl': 24,
      '3xl': 28,
      '4xl': 32,
      '5xl': 36,
    },
    fontWeight: {
      normal: '400',
      medium: '500',
      semiBold: '600',
      bold: '700',
    },
    lineHeight: {
      tight: 1.2,
      normal: 1.4,
      relaxed: 1.6,
    },
  },
  
  // Spacing scale
  spacing: {
    0: 0,
    1: 4,
    2: 8,
    3: 12,
    4: 16,
    5: 20,
    6: 24,
    7: 28,
    8: 32,
    10: 40,
    12: 48,
    16: 64,
    20: 80,
    24: 96,
    32: 128,
  },
  
  // Border radius scale
  borderRadius: {
    none: 0,
    xs: 2,
    sm: 4,
    md: 6,
    lg: 8,
    xl: 12,
    '2xl': 16,
    '3xl': 20,
    '4xl': 24,
    full: 9999,
  },
  
  // Shadow scale
  shadow: {
    sm: {
      shadowOffset: { width: 0, height: 1 },
      shadowOpacity: 0.05,
      shadowRadius: 2,
      elevation: 1,
    },
    md: {
      shadowOffset: { width: 0, height: 2 },
      shadowOpacity: 0.1,
      shadowRadius: 4,
      elevation: 3,
    },
    lg: {
      shadowOffset: { width: 0, height: 4 },
      shadowOpacity: 0.15,
      shadowRadius: 8,
      elevation: 5,
    },
    xl: {
      shadowOffset: { width: 0, height: 8 },
      shadowOpacity: 0.2,
      shadowRadius: 16,
      elevation: 8,
    },
  },
  
  // Animation durations
  animation: {
    duration: {
      fast: 150,
      normal: 250,
      slow: 350,
    },
    easing: {
      default: 'ease',
      easeIn: 'ease-in',
      easeOut: 'ease-out',
      easeInOut: 'ease-in-out',
    },
  },
  
  // Component-specific sizes
  components: {
    button: {
      height: {
        sm: 32,
        md: 40,
        lg: 48,
        xl: 56,
      },
      paddingHorizontal: {
        sm: 12,
        md: 16,
        lg: 20,
        xl: 24,
      },
    },
    input: {
      height: {
        sm: 36,
        md: 44,
        lg: 52,
      },
      paddingHorizontal: 12,
    },
    avatar: {
      size: {
        xs: 24,
        sm: 32,
        md: 40,
        lg: 48,
        xl: 56,
        '2xl': 64,
      },
    },
    icon: {
      size: {
        xs: 12,
        sm: 16,
        md: 20,
        lg: 24,
        xl: 28,
      },
    },
  },
  
  // Layout constants
  layout: {
    screenPadding: 20,
    cardPadding: 16,
    sectionSpacing: 24,
    headerHeight: 56,
    tabBarHeight: 80,
  },
  
  // Camera and gallery specific
  camera: {
    captureButtonSize: 80,
    captureButtonInner: 60,
    controlsHeight: 120,
    previewAspectRatio: 4 / 3,
  },
  
  gallery: {
    gridSpacing: 8,
    itemAspectRatio: 1,
    thumbnailSize: 120,
  },
} as const;

export type Globals = typeof globals;