/**
 * BNA UI Theme Provider
 * Provides theme context throughout the app
 */

import React, { createContext, useContext, useEffect, useState } from 'react';
import { Appearance, ColorSchemeName } from 'react-native';
import { colors, type Colors } from './colors';
import { globals, type Globals } from './globals';

export type ColorScheme = 'light' | 'dark';

export interface Theme {
  colors: Colors;
  globals: Globals;
  colorScheme: ColorScheme;
}

interface ThemeContextType {
  theme: Theme;
  colorScheme: ColorScheme;
  setColorScheme: (scheme: ColorScheme) => void;
  toggleColorScheme: () => void;
}

const ThemeContext = createContext<ThemeContextType | undefined>(undefined);

interface ThemeProviderProps {
  children: React.ReactNode;
  defaultColorScheme?: ColorScheme;
}

export function ThemeProvider({ 
  children, 
  defaultColorScheme 
}: ThemeProviderProps) {
  const [colorScheme, setColorScheme] = useState<ColorScheme>(() => {
    if (defaultColorScheme) return defaultColorScheme;
    
    const systemColorScheme = Appearance.getColorScheme();
    return systemColorScheme === 'dark' ? 'dark' : 'light';
  });

  const theme: Theme = {
    colors: colors[colorScheme],
    globals,
    colorScheme,
  };

  const toggleColorScheme = () => {
    setColorScheme(prev => prev === 'light' ? 'dark' : 'light');
  };

  // Listen to system color scheme changes
  useEffect(() => {
    const subscription = Appearance.addChangeListener(({ colorScheme: systemColorScheme }) => {
      if (!defaultColorScheme) {
        setColorScheme(systemColorScheme === 'dark' ? 'dark' : 'light');
      }
    });

    return () => subscription?.remove();
  }, [defaultColorScheme]);

  const value = {
    theme,
    colorScheme,
    setColorScheme,
    toggleColorScheme,
  };

  return (
    <ThemeContext.Provider value={value}>
      {children}
    </ThemeContext.Provider>
  );
}

export function useTheme(): ThemeContextType {
  const context = useContext(ThemeContext);
  if (!context) {
    throw new Error('useTheme must be used within a ThemeProvider');
  }
  return context;
}

export function useThemeColor(
  lightColor?: string,
  darkColor?: string
): string {
  const { colorScheme } = useTheme();
  
  if (lightColor && darkColor) {
    return colorScheme === 'dark' ? darkColor : lightColor;
  }
  
  return lightColor || darkColor || '#000000';
}