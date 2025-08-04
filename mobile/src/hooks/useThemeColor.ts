/**
 * Theme color hook
 * Provides easy access to theme colors
 */

import { useTheme } from '../theme/theme-provider';
import type { ColorName } from '../theme/colors';

export function useThemeColor(colorName?: ColorName, lightColor?: string, darkColor?: string) {
  const { theme, colorScheme } = useTheme();
  
  // If specific color name is provided, use theme colors
  if (colorName) {
    return theme.colors[colorName];
  }
  
  // If light/dark colors are provided, use them based on scheme
  if (lightColor && darkColor) {
    return colorScheme === 'dark' ? darkColor : lightColor;
  }
  
  // Default fallback
  return lightColor || darkColor || theme.colors.text;
}