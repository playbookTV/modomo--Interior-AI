/**
 * Color scheme hook
 * Provides access to current color scheme
 */

import { useTheme } from '../theme/theme-provider';

export function useColorScheme() {
  const { colorScheme, setColorScheme, toggleColorScheme } = useTheme();
  
  return {
    colorScheme,
    setColorScheme,
    toggleColorScheme,
    isDark: colorScheme === 'dark',
    isLight: colorScheme === 'light',
  };
}