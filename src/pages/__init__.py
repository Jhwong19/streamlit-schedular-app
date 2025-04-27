# Import all page functions to make them available from this package
from src.pages._home_page import home_page
from src.pages._about_page import about_page
from src.pages._contact_page import contact_page
from src.pages._map_page import map_page
from src.pages._optimize_page import optimize_page

__all__ = ['home_page', 'about_page', 'contact_page', 'map_page', 'optimize_page']