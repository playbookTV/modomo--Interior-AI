/**
 * Mode toggle hook
 * Provides theme toggle functionality
 */

import { useColorScheme } from './useColorScheme';

export function useModeToggle() {
  const { toggleColorScheme, isDark, isLight } = useColorScheme();
  
  return {
    toggle: toggleColorScheme,
    isDark,
    isLight,
    mode: isDark ? 'dark' : 'light',
  };
}