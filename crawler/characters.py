# not doing that anymore
import scrapy
from bs4 import BeautifulSoup
class CharacterSpider(scrapy.Spider):
    name = 'charactersSpider'
    start_urls = ['https://strangerthings.fandom.com/wiki/Category:Characters']

    def parse(self, response):
        for href in response.css('.category-page__members')[0].css("a::attr(href)").extract():
            extracted_data = scrapy.Request("https://strangerthings.fandom.com" + href,
                           callback=self.parse_character)
            yield extracted_data

        for next_page in response.css('a.category-page__pagination-next'):
            yield response.follow(next_page, self.parse)
            
    def parse_character(self, response):
        character_name = response.css("span.mw-page-title-main::text").extract()[0]
        character_name = character_name.strip()
        
        div_selector = response.css("div.mw-content-ltr.mw-parser-output")[0]
        div_html = div_selector.extract()
        
        soup = BeautifulSoup(div_html).find('div')
        #get affiliation
        char_affiliation = ''
        if soup and soup.find('aside'):
            aside = soup.find('aside')
            for cell in aside.find_all('div',{'class': 'pi-data'} ):
                if cell.find('h3'):
                    cell_name = cell.find('h3').text.strip()
                    if cell_name == 'Affiliation':
                        char_affiliation = cell.find('div').text.strip()
        #get description
        if soup and soup.find('aside'):
            soup.find('aside').decompose()
        toc_div = soup.find('div', class_='toc')
        if not toc_div:
            toc_div = soup.find('table', class_='mw-collapsible')
        
        # extract content above the 'toc' div or 'table.mw-collapsible'
        char_description = ''
        for element in soup.find_all():
            if element == toc_div:
                break
            char_description += element.get_text(separator=' ', strip=True) + ' '
        
        char_description = char_description.strip()
        
        return dict (
            character_name = character_name,
            char_affiliation = char_affiliation,
            char_description = char_description
        )                          